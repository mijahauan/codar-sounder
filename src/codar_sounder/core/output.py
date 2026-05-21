"""JSON-Lines writer for codar-sounder ionospheric records.

One file per (radiod_id, station_id, UTC date) triple at::

    /var/lib/codar-sounder/<radiod_id>/<station>/<YYYY>/<MM>/<DD>.jsonl

Each line is a self-contained JSON object: timestamp, group range,
virtual height, equivalent vertical frequency, SNR, layer label, plus
the geometry that produced the inversion.  Downstream consumers
(HamSCI archive, TID/space-weather post-processors) parse one record
per detected peak.

Schema (v0.5):
    timestamp                        ISO-8601 UTC
    client                           "codar-sounder"
    radiod_id, radiod_status_dns     identity
    station_id                       transmitter id (e.g. "DUCK")
    oblique_freq_hz, sweep_rate_hz_per_s, coherent_seconds   measurement context
    peak_index                       0-based rank within the CPI (0 = strongest)
    peak_count                       total peaks reported for this CPI
    mode_layer                       "E" / "F1" / "F2" / "F2_extreme" / "below_E"
    snr_db                           peak vs. window-median (dB)
    group_range_km, ground_distance_km
    virtual_height_km (+ uncertainty)
    equivalent_vertical_freq_mhz (+ uncertainty)
    takeoff_zenith_deg
    s4_index                         amplitude scintillation, ITU-R P.531 (≥0;
                                     may exceed 1.0 for saturated scintillation)
    s4_severity                      "weak"/"moderate"/"strong"/"unknown"
    sigma_phi_rad                    phase scintillation, radians (≥0)
    sigma_phi_severity               same bins as s4_severity
    scintillation_event              true iff S4 ≥ 0.3 or σ_φ ≥ 0.2
    scintillation_confidence         0..1, saturates at 30 slow-time samples
    scintillation_samples            slow-time samples retained after MAD reject (v0.5)
    scintillation_outliers_rejected  count dropped by MAD filter (v0.5.1)
    mode_doppler_hz                  linear-Doppler shift removed before σ_φ
    sigma_phi_linear_rad             σ_φ from linear detrend (v0.6 diagnostic)
    sigma_phi_quadratic_rad          σ_φ from quadratic detrend (= sigma_phi_rad)
    sigma_phi_underfit_ratio         linear/quadratic ratio; >>1 → phase curvature
    dechirp_sweeps_rejected          sweeps zeroed by per-sweep MAD pre-filter (v0.6.1)
    n_hops                           ionospheric hop count selected by invert() (v0.7)

  Scintillation fields are computed per-CPI per-peak from the
  pre-Doppler-FFT range-spectrum column at the peak's range bin — one
  slow-time sample per sweep, M samples per CPI (60 at default
  CPI=60s, SRF=1Hz).  See ``codar_sounder.core.scintillation``.

Atomic-ish writes: each record is written with a trailing newline and
``flush()``.  We don't fsync per-record (would dominate I/O cost on
hosts with rotational disks) — at most one record can be lost on crash.

This file remains the canonical L1 artefact.  The CONTRACT v0.6 §17
HamSCI sink (codar.spots) is *additive* — written from the daemon
in parallel via :class:`sigmond.hamsci_sink.Writer`, never instead of.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Optional

from codar_sounder.core.invert import IonosphericFix
from codar_sounder.core.scintillation import ScintillationResult
from codar_sounder.core.trace import TraceDetection

log = logging.getLogger(__name__)


class JsonlWriter:
    """Daily-rotated JSON-Lines writer keyed on (radiod_id, station)."""

    def __init__(self, output_root: Path, radiod_id: str, station_id: str):
        self.output_root = Path(output_root)
        self.radiod_id = radiod_id
        self.station_id = station_id
        self._current_date: Optional[str] = None
        self._fh: Optional[IO[str]] = None

    def _path_for(self, ts: datetime) -> Path:
        d = ts.astimezone(timezone.utc)
        return (
            self.output_root
            / self.radiod_id
            / self.station_id
            / f"{d.year:04d}"
            / f"{d.month:02d}"
            / f"{d.day:02d}.jsonl"
        )

    def _ensure_open(self, ts: datetime) -> IO[str]:
        d = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if self._current_date != d:
            self.close()
            path = self._path_for(ts)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self._current_date = d
            log.info("opened %s", path)
        assert self._fh is not None
        return self._fh

    def write(
        self,
        timestamp: datetime,
        fix: IonosphericFix,
        detection: TraceDetection,
        scintillation: ScintillationResult,
        *,
        radiod_status_dns: str,
        oblique_freq_hz: int,
        coherent_seconds: float,
        sweep_rate_hz_per_s: float,
        peak_index: int = 0,
        peak_count: int = 1,
        dechirp_sweeps_rejected: int = 0,
    ) -> Path:
        """Write one record and return the path it landed in.

        ``peak_index`` is the 0-based rank of this peak among the CPI's
        detections (0 = strongest); ``peak_count`` is the total number
        of peaks reported for this CPI.  When the daemon emits one
        record per detected peak, downstream consumers use these to
        regroup peaks back into their parent CPI.

        ``scintillation`` carries the ITU-R P.531 S4 / σ_φ indices
        computed on this peak's slow-time vector (see
        :mod:`codar_sounder.core.scintillation`).  The severity strings
        are canonical: ``s4_index`` and ``sigma_phi_rad`` are written
        full-precision (no rounding) so a downstream consumer
        reproducing the severity from the numeric value can do so
        deterministically across the bin boundaries.
        """
        record: dict[str, Any] = {
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            "client": "codar-sounder",
            "radiod_id": self.radiod_id,
            "radiod_status_dns": radiod_status_dns,
            "station_id": self.station_id,
            "oblique_freq_hz": oblique_freq_hz,
            "sweep_rate_hz_per_s": sweep_rate_hz_per_s,
            "coherent_seconds": coherent_seconds,
            "peak_index": peak_index,
            "peak_count": peak_count,
            "mode_layer": fix.mode_layer,
            "snr_db": round(detection.snr_db, 2),
            "group_range_km": round(fix.group_range_km, 2),
            "ground_distance_km": round(fix.ground_distance_km, 2),
            "virtual_height_km": round(fix.virtual_height_km, 2),
            "virtual_height_uncertainty_km": round(
                fix.virtual_height_uncertainty_km, 2
            ),
            "equivalent_vertical_freq_mhz": round(
                fix.equivalent_vertical_freq_mhz, 4
            ),
            "equivalent_vertical_freq_uncertainty_mhz": round(
                fix.equivalent_vertical_freq_uncertainty_mhz, 4
            ),
            "takeoff_zenith_deg": round(fix.takeoff_zenith_deg, 2),
            # Scintillation indices — full precision on the numerics so
            # readers can reproduce the severity bins deterministically.
            "s4_index": float(scintillation.s4_index),
            "s4_severity": scintillation.s4_severity,
            "sigma_phi_rad": float(scintillation.sigma_phi_rad),
            "sigma_phi_severity": scintillation.sigma_phi_severity,
            "scintillation_event": bool(scintillation.scintillation_event),
            "scintillation_confidence": round(scintillation.confidence, 3),
            "scintillation_samples": int(scintillation.n_samples),
            "scintillation_outliers_rejected":
                int(scintillation.n_outliers_rejected),
            "mode_doppler_hz": round(scintillation.mode_doppler_hz, 4),
            "n_hops": int(fix.n_hops),
            # v0.6 diagnostics — full precision so the linear/quadratic
            # ratio is reproducible by downstream readers.
            "sigma_phi_linear_rad": float(scintillation.sigma_phi_linear_rad),
            "sigma_phi_quadratic_rad":
                float(scintillation.sigma_phi_quadratic_rad),
            "sigma_phi_underfit_ratio":
                round(scintillation.sigma_phi_underfit_ratio, 3),
            "dechirp_sweeps_rejected": int(dechirp_sweeps_rejected),
        }
        fh = self._ensure_open(timestamp)
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()
        return self._path_for(timestamp)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._current_date = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

"""Sounder daemon — v0.4.

End-to-end pipeline.  One ``SounderDaemon`` per radiod, fanning each
CPI out across every configured ``[[radiod.transmitter]]``:

    radiod IQ channel (one subscription per radiod)
        ↓ ka9q-python RadiodStream (or synthetic fallback)
    contiguous CPI of complex64 samples
        ↓ per-transmitter pipeline:
            ↓ core.dechirp (replica matched to this TX's sweep)
        range-Doppler matrix
            ↓ core.dechirp.range_profile + positive_range_window
        range profile (positive ranges only)
            ↓ core.trace.find_f_region_peaks  (with ground-clutter mask)
        list[TraceDetection] — every local maximum above SNR threshold,
                              up to ``max_peaks`` (default 4) per CPI,
                              sorted SNR-descending
            ↓ for each peak:
                ↓ core.invert.invert  (with classify_layer)
            IonosphericFix (virtual_height, fv, uncertainty, mode_layer)
                ↓ JSONL: core.output.JsonlWriter (canonical L1 artefact,
                  Kaeppler-compatible Zenodo schema; one record per peak)
                ↓ HamSCI sink (CONTRACT v0.6 §17, additive when configured):
                  sigmond.hamsci_sink.Writer → codar.spots, one row per peak

Per-CPI failures (no peak detected, low SNR, unphysical geometry on
one of several peaks, sink unreachable) emit a warning and continue —
never crash the service.  The sink path failing never blocks JSONL.

JSONL output layout:
    /var/lib/codar-sounder/<radiod>/<station>/<YYYY>/<MM>/<DD>.jsonl
    one record per detected peak, ranked by ``peak_index`` /
    ``peak_count`` so a downstream consumer can regroup peaks back into
    their parent CPI.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from codar_sounder.config import haversine_km, transmitters
from codar_sounder.core.dechirp import dechirp, positive_range_window, range_profile
from codar_sounder.core.invert import invert, group_range_resolution_km
from codar_sounder.core.output import JsonlWriter
from codar_sounder.core.stream import make_iq_source
from codar_sounder.core.trace import (
    DEFAULT_MAX_PEAKS,
    GroundClutterMask,
    find_f_region_peaks,
)

log = logging.getLogger(__name__)


def _sd_notify(message: str) -> None:
    """Minimal sd_notify implementation — no python-systemd dependency."""
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(sock_path)
            s.sendall(message.encode("utf-8"))
    except OSError as exc:
        log.warning("sd_notify failed (%s): %s", message, exc)


class _TransmitterPipeline:
    """One CODAR-transmitter's end-to-end processing pipeline.

    A daemon manages one of these per ``[[radiod.transmitter]]`` block.
    Each has its own clutter mask, JsonlWriter, and dechirp parameters.
    """

    def __init__(
        self,
        *,
        receiver_lat: float,
        receiver_lon: float,
        radiod_id: str,
        radiod_status_dns: str,
        tx_config: dict,
        output_dir: Path,
        sample_rate_hz: float,
        coherent_seconds: float,
        range_min_km: float,
        range_max_km: float,
        snr_threshold_db: float,
        clutter_window: int = 20,
        max_peaks_per_cpi: int = DEFAULT_MAX_PEAKS,
        ch_writer=None,
        host_call: str = "",
        host_grid: str = "",
        processing_version: str = "",
    ):
        self.station_id = tx_config["id"]
        self.tx_config = tx_config
        self.center_freq_hz = int(tx_config["center_freq_hz"])
        self.sweep_rate_hz_per_s = float(tx_config["sweep_rate_hz_per_s"])
        self.sweep_bw_hz = float(tx_config["sweep_bw_hz"])
        self.sweep_repetition_hz = float(tx_config["sweep_repetition_hz"])
        # Optional TDMA sweep-start phase, in samples within one period.
        # Operator-supplied (or `codar-sounder tdma-scan` discovered);
        # default 0 = v0.2 behaviour (no phase alignment).  Co-band TXs
        # at the same frequency cannot be distinguished without this
        # — see core/tdma.py.
        self.tdma_offset_samples = int(tx_config.get("tdma_offset_samples", 0))

        self.ground_distance_km = haversine_km(
            receiver_lat, receiver_lon,
            float(tx_config["tx_lat_deg"]), float(tx_config["tx_lon_deg"]),
        )
        self.group_range_uncertainty_km = group_range_resolution_km(self.sweep_bw_hz)
        self.radiod_id = radiod_id
        self.radiod_status_dns = radiod_status_dns
        self.sample_rate_hz = sample_rate_hz
        self.coherent_seconds = coherent_seconds
        self.range_min_km = range_min_km
        self.range_max_km = range_max_km
        self.snr_threshold_db = snr_threshold_db
        self.max_peaks_per_cpi = max_peaks_per_cpi
        # Optional second sink for CONTRACT v0.6 §17 — local HamSCI
        # sink (SQLite store-and-forward queue).  When present, each
        # per-peak record is also written to the codar.spots table.
        # None when the sink isn't configured; set in
        # SounderDaemon.__init__ from sigmond.hamsci_sink.Writer.
        self.ch_writer = ch_writer
        self.host_call = host_call
        self.host_grid = host_grid
        self.processing_version = processing_version

        self.clutter_mask = GroundClutterMask(window=clutter_window)
        self.writer = JsonlWriter(
            output_root=output_dir,
            radiod_id=radiod_id,
            station_id=self.station_id,
        )
        log.info(
            "pipeline init: tx=%s freq=%d Hz κ=%.1f Hz/s BW=%.0f Hz "
            "TX-RX dist=%.1f km ΔP_resolution=%.1f km tdma_offset=%d samples",
            self.station_id, self.center_freq_hz, self.sweep_rate_hz_per_s,
            self.sweep_bw_hz, self.ground_distance_km,
            self.group_range_uncertainty_km, self.tdma_offset_samples,
        )

    def process_cpi(self, rx_samples, cpi_start_utc: datetime) -> Optional[Path]:
        """Run one CPI through dechirp → trace → invert → write.

        Emits one JSONL record per detected peak (high-ray, low-ray,
        E-layer, etc.); each record is layer-classified.  Returns the
        path of the most-recent write (if any peak was successfully
        recorded), else ``None``.  Never raises — per-CPI failures are
        logged and swallowed so the daemon can keep running.
        """
        try:
            result = dechirp(
                rx_samples,
                sample_rate_hz=self.sample_rate_hz,
                sweep_rate_hz_per_s=self.sweep_rate_hz_per_s,
                sweep_repetition_hz=self.sweep_repetition_hz,
                phase_offset_samples=self.tdma_offset_samples,
            )
        except Exception as exc:
            log.warning("[%s] dechirp failed: %s", self.station_id, exc)
            return None

        ranges_km, profile = positive_range_window(result, range_profile(result))

        try:
            detections = find_f_region_peaks(
                profile, ranges_km,
                range_min_km=self.range_min_km,
                range_max_km=self.range_max_km,
                snr_threshold_db=self.snr_threshold_db,
                clutter_mask=self.clutter_mask,
                max_peaks=self.max_peaks_per_cpi,
            )
        except Exception as exc:
            log.warning("[%s] trace extraction failed: %s", self.station_id, exc)
            return None

        if not detections:
            log.debug("[%s] no F-region peak above %.1f dB this CPI",
                      self.station_id, self.snr_threshold_db)
            return None

        # CPI timestamp comes from the iq_source (RTP-derived + authority
        # offset for radiod path; wall-clock for synthetic).  Per
        # METROLOGY.md §4.5 RTP-reference invariant — the recorder does
        # not consult wall clock for data labels.
        ts = cpi_start_utc
        last_path: Optional[Path] = None
        peak_count = len(detections)
        for peak_index, detection in enumerate(detections):
            try:
                fix = invert(
                    group_range_km=detection.group_range_km,
                    ground_distance_km=self.ground_distance_km,
                    oblique_freq_mhz=self.center_freq_hz / 1e6,
                    group_range_uncertainty_km=self.group_range_uncertainty_km,
                )
            except ValueError as exc:
                # group_range < ground_distance — geometrically impossible.
                # Short-range clutter the mask missed; skip this peak,
                # keep processing the rest.
                log.warning(
                    "[%s] inversion rejected peak %d/%d at %.0f km "
                    "(TX-RX %.0f km): %s",
                    self.station_id, peak_index, peak_count,
                    detection.group_range_km, self.ground_distance_km, exc,
                )
                continue

            last_path = self.writer.write(
                timestamp=ts,
                fix=fix,
                detection=detection,
                radiod_status_dns=self.radiod_status_dns,
                oblique_freq_hz=self.center_freq_hz,
                coherent_seconds=self.coherent_seconds,
                sweep_rate_hz_per_s=self.sweep_rate_hz_per_s,
                peak_index=peak_index,
                peak_count=peak_count,
            )

            # CONTRACT v0.6 §17 — additive write to the local HamSCI
            # sink (SQLite store-and-forward queue).  JSONL above
            # remains the canonical L1 artefact (Kaeppler-compatible
            # Zenodo schema).  Sink path failure is non-fatal.
            if self.ch_writer is not None:
                row = self._ch_row_for(
                    timestamp=ts, detection=detection, fix=fix,
                    peak_index=peak_index, peak_count=peak_count,
                )
                try:
                    self.ch_writer.insert([row])
                except Exception as exc:
                    log.warning(
                        "[%s] sink insert failed for peak %d/%d: %s",
                        self.station_id, peak_index, peak_count, exc,
                    )

            log.info(
                "[%s] peak %d/%d %s P=%.0f km h'=%.0f±%.0f km "
                "fv=%.3f±%.3f MHz SNR=%.1f dB",
                self.station_id, peak_index, peak_count, fix.mode_layer,
                fix.group_range_km, fix.virtual_height_km,
                fix.virtual_height_uncertainty_km,
                fix.equivalent_vertical_freq_mhz,
                fix.equivalent_vertical_freq_uncertainty_mhz,
                detection.snr_db,
            )
        return last_path

    def _ch_row_for(
        self, *, timestamp, detection, fix, peak_index: int, peak_count: int,
    ) -> dict:
        """Build a row for the codar.spots HamSCI sink table.

        Numeric fields keep full precision (no rounding) — the
        analytics side benefits from the extra digits.
        """
        return {
            "time":               timestamp,
            "host_call":          self.host_call,
            "host_grid":          self.host_grid,
            "radiod_id":          self.radiod_id,
            "instance":           self.radiod_id,
            "processing_version": self.processing_version,
            "station_id":         self.station_id,
            "oblique_freq_hz":    int(self.center_freq_hz),
            "sweep_rate_hz_per_s": float(self.sweep_rate_hz_per_s),
            "coherent_seconds":   float(self.coherent_seconds),
            "peak_index":         int(peak_index),
            "peak_count":         int(peak_count),
            "mode_layer":         fix.mode_layer,
            "snr_db":             float(detection.snr_db),
            "group_range_km":     float(fix.group_range_km),
            "ground_distance_km": float(fix.ground_distance_km),
            "virtual_height_km":  float(fix.virtual_height_km),
            "virtual_height_uncertainty_km":
                float(fix.virtual_height_uncertainty_km),
            "equivalent_vertical_freq_mhz":
                float(fix.equivalent_vertical_freq_mhz),
            "equivalent_vertical_freq_uncertainty_mhz":
                float(fix.equivalent_vertical_freq_uncertainty_mhz),
            "takeoff_zenith_deg": float(fix.takeoff_zenith_deg),
        }

    def close(self) -> None:
        self.writer.close()


class SounderDaemon:
    """One daemon per radiod.  Iterates the IQ source, fans each CPI
    out across the configured transmitters, and writes JSONL records.

    Key design choices (see plan §"One daemon instance per radiod"):

      * One IQ subscription per radiod — every transmitter in the same
        band is dechirped from the *same* CPI in parallel.  TDMA-offset
        CODAR transmitters that share a band get separate dechirp
        passes with their own sweep parameters.

      * The clutter mask is per-transmitter, not per-radiod, because
        different sweep rates produce different ground-clutter
        signatures.

      * The IQ source backend is selected by ``make_iq_source`` —
        radiod when ka9q-python is importable, synthetic fallback
        otherwise (with a loud log warning).
    """

    def __init__(self, config: dict, radiod_block: dict):
        self.config = config
        self.radiod = radiod_block
        self.station = config.get("station", {})
        self.paths = config.get("paths", {})
        self.processing = config.get("processing", {})
        self._stopped = threading.Event()

        self.radiod_id = radiod_block.get("id", "default")
        self.channel_name = radiod_block.get("channel_name", "codar")
        self.status_dns = radiod_block.get("status_dns", "")

        # Sample rate is implicitly set by the radiod fragment
        # (etc/radiod-fragment.conf samprate field); v0.2 reads it from
        # the [processing] block as `sample_rate_hz`, defaulting to
        # 64000 (matches the 4.5 MHz CODAR fragment sample rate).
        self.sample_rate_hz = float(self.processing.get("sample_rate_hz", 64000))
        self.coherent_seconds = float(self.processing.get("coherent_seconds", 60))
        self.range_min_km = float(self.processing.get("range_min_km", 200))
        self.range_max_km = float(self.processing.get("range_max_km", 800))
        self.snr_threshold_db = float(self.processing.get("snr_threshold_db", 5.0))

        # Crash-safe channel cleanup via radiod's LIFETIME tag (ka9q-python
        # 7c6af73+, radiod 0f8b622+).  Default: 2 × CPI seconds at the
        # radiod default frame rate (50 Hz at 20 ms blocktime).  Each CPI
        # yield refreshes the lifetime, so a missed CPI is tolerated; if
        # the daemon dies, the channel self-destructs within 2 × CPI.
        # Operators can override to 0 (= explicit infinite, matches
        # pre-v0.4 behaviour) or any positive frame count.  ``None``
        # would be ambiguous in TOML (omitting the key gives the default).
        self.radiod_lifetime_frames: Optional[int] = self.processing.get(
            "radiod_lifetime_frames",
            int(round(2 * self.coherent_seconds * 50)),
        )
        if self.radiod_lifetime_frames is not None:
            self.radiod_lifetime_frames = int(self.radiod_lifetime_frames)
            if self.radiod_lifetime_frames < 0:
                raise ValueError(
                    f"processing.radiod_lifetime_frames must be ≥ 0; got "
                    f"{self.radiod_lifetime_frames}"
                )

        receiver_lat = float(self.station.get("receiver_lat", 0.0))
        receiver_lon = float(self.station.get("receiver_lon", 0.0))
        output_dir = Path(self.paths.get("output_dir", "/var/lib/codar-sounder"))

        # CONTRACT v0.6 §17 — local HamSCI sink (SQLite store-and-
        # forward queue).  One Writer per daemon, shared across all
        # transmitter pipelines (they all write to codar.spots).
        # Returns a no-op writer when no sink path is configured and
        # /var/lib/sigmond isn't writable, so this is safe in the
        # standalone (no-sigmond) case too.  Module-not-found means
        # sigmond.hamsci_sink isn't installed — log and stay file-only.
        self.ch_writer = None
        try:
            from sigmond.hamsci_sink import Writer as _HamsciWriter  # type: ignore[import-not-found]
            self.ch_writer = _HamsciWriter.from_env(
                table="spots", mode="codar",
                schema_version=1, batch_rows=200,
            )
            log.info(
                "sink writer health=%s (database=%s)",
                self.ch_writer.health, self.ch_writer.database,
            )
        except ImportError:
            log.debug("sigmond.hamsci_sink not available; sink path disabled")
        except Exception as exc:
            log.warning("sink writer init failed (%s); JSONL path unaffected", exc)

        host_call = str(self.station.get("callsign", ""))
        host_grid = str(self.station.get("grid_square", ""))
        proc_version = self._processing_version_string()

        # All transmitters in this radiod's band share one IQ
        # subscription; the daemon dechirps each separately.
        self.pipelines: list[_TransmitterPipeline] = []
        for tx in transmitters(radiod_block):
            self.pipelines.append(_TransmitterPipeline(
                receiver_lat=receiver_lat,
                receiver_lon=receiver_lon,
                radiod_id=self.radiod_id,
                radiod_status_dns=self.status_dns,
                tx_config=tx,
                output_dir=output_dir,
                sample_rate_hz=self.sample_rate_hz,
                coherent_seconds=self.coherent_seconds,
                range_min_km=self.range_min_km,
                range_max_km=self.range_max_km,
                snr_threshold_db=self.snr_threshold_db,
                ch_writer=self.ch_writer,
                host_call=host_call,
                host_grid=host_grid,
                processing_version=proc_version,
            ))

        # Pick the dominant sweep rate / SRF for the synthetic fallback
        # (one transmitter's worth of synthetic IQ — better than no IQ).
        first_tx = self.pipelines[0] if self.pipelines else None
        force_synthetic = bool(self.processing.get("force_synthetic", False))

        # Geometry-aware default for the synthetic-fallback target group
        # range: place the notional ionospheric reflector at a typical
        # F-region virtual height (250 km) above the path midpoint, so
        # the smoke-test inversion produces self-consistent records on
        # any host regardless of TX-RX distance.
        # P = sqrt(D² + (2h)²)  with h = 250 km
        default_h_km = float(self.processing.get(
            "synthetic_target_height_km", 250.0
        ))
        if first_tx is not None:
            default_synth_p = (
                first_tx.ground_distance_km ** 2 + (2 * default_h_km) ** 2
            ) ** 0.5
        else:
            default_synth_p = 500.0
        synth_p = float(self.processing.get(
            "synthetic_target_group_range_km", default_synth_p
        ))

        # 0 in config means "explicit infinite" — pass None to ka9q-python
        # so the LIFETIME tag is omitted (and per-CPI refresh is skipped).
        # Any positive value enables crash-safe cleanup.
        lifetime_for_source: Optional[int] = (
            None if self.radiod_lifetime_frames in (None, 0)
            else self.radiod_lifetime_frames
        )
        self.iq_source = make_iq_source(
            radiod_status_dns=self.status_dns,
            channel_name=self.channel_name,
            sample_rate_hz=self.sample_rate_hz,
            cpi_seconds=self.coherent_seconds,
            sweep_rate_hz_per_s=first_tx.sweep_rate_hz_per_s if first_tx else -25733.913,
            sweep_repetition_hz=first_tx.sweep_repetition_hz if first_tx else 1.0,
            center_freq_hz=first_tx.center_freq_hz if first_tx else 4_537_180.0,
            preset=str(radiod_block.get("preset", "iq")),
            fallback_target_group_range_km=synth_p,
            force_synthetic=force_synthetic,
            lifetime_frames=lifetime_for_source,
        )

    def run(self) -> None:
        if not self.pipelines:
            log.error("no transmitters configured for radiod %s — exiting",
                      self.radiod_id)
            return

        log.info(
            "starting daemon: radiod=%s channel=%s sample_rate=%d Hz "
            "CPI=%.1f s transmitters=%s",
            self.radiod_id, self.channel_name, int(self.sample_rate_hz),
            self.coherent_seconds,
            [p.station_id for p in self.pipelines],
        )
        _sd_notify(
            f"READY=1\nSTATUS=processing {len(self.pipelines)} CODAR txs "
            f"on radiod {self.radiod_id}"
        )

        try:
            for cpi_samples, cpi_start_utc in self.iq_source:
                if self._stopped.is_set():
                    break
                # Fan out: every transmitter sees the same IQ; each
                # dechirps with its own sweep params, finds its own peak.
                for pipeline in self.pipelines:
                    pipeline.process_cpi(cpi_samples, cpi_start_utc)
                _sd_notify("WATCHDOG=1")
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.close()

    def stop(self) -> None:
        self._stopped.set()
        if hasattr(self.iq_source, "stop"):
            self.iq_source.stop()

    def close(self) -> None:
        for pipeline in self.pipelines:
            pipeline.close()
        if self.ch_writer is not None:
            try:
                self.ch_writer.close()
            except Exception as exc:
                log.warning("sink writer close failed: %s", exc)

    def _processing_version_string(self) -> str:
        """Return ``<package_version>+<git_short>`` for the CH ``processing_version``."""
        try:
            from importlib.metadata import version as pkg_version
            base = pkg_version("codar-sounder")
        except Exception:
            base = "0.4.0"
        try:
            from codar_sounder.version import GIT_INFO as _GIT
            short = (_GIT or {}).get("short", "")
        except Exception:
            short = ""
        return f"{base}+{short}" if short else base

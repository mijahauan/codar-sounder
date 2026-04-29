"""Sounder daemon — v0.2.

End-to-end pipeline for one (radiod, CODAR-transmitter) pair:

    radiod IQ channel
        ↓ ka9q-python RadiodStream (or synthetic fallback)
    contiguous CPI of complex64 samples
        ↓ core.dechirp
    range-Doppler matrix
        ↓ core.dechirp.range_profile + positive_range_window
    range profile (positive ranges only)
        ↓ core.trace.find_f_region_peak (with ground-clutter mask)
    F-region peak (group_range, snr) — or None
        ↓ core.invert.invert
    IonosphericFix (virtual_height, equivalent_vertical_freq, uncertainty)
        ↓ core.output.JsonlWriter
    /var/lib/codar-sounder/<radiod>/<station>/<YYYY>/<MM>/<DD>.jsonl

One ``SounderDaemon`` instance per (radiod, transmitter) pair.  The
daemon's ``run()`` loop is a thin orchestrator: receive CPI, dechirp,
detect, invert, write.  Per-CPI failures (no peak detected, low SNR,
unphysical geometry) emit a warning and continue — never crash the
service.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from codar_sounder.config import haversine_km, transmitters
from codar_sounder.core.dechirp import dechirp, positive_range_window, range_profile
from codar_sounder.core.invert import invert, group_range_resolution_km
from codar_sounder.core.output import JsonlWriter
from codar_sounder.core.stream import make_iq_source
from codar_sounder.core.trace import GroundClutterMask, find_f_region_peak

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

    def process_cpi(self, rx_samples) -> Optional[Path]:
        """Run one CPI through dechirp → trace → invert → write.

        Returns the path written (if a peak was detected and inversion
        succeeded), else ``None``.  Never raises — per-CPI failures are
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
            detection = find_f_region_peak(
                profile, ranges_km,
                range_min_km=self.range_min_km,
                range_max_km=self.range_max_km,
                snr_threshold_db=self.snr_threshold_db,
                clutter_mask=self.clutter_mask,
            )
        except Exception as exc:
            log.warning("[%s] trace extraction failed: %s", self.station_id, exc)
            return None

        if detection is None:
            log.debug("[%s] no F-region peak above %.1f dB this CPI",
                      self.station_id, self.snr_threshold_db)
            return None

        try:
            fix = invert(
                group_range_km=detection.group_range_km,
                ground_distance_km=self.ground_distance_km,
                oblique_freq_mhz=self.center_freq_hz / 1e6,
                group_range_uncertainty_km=self.group_range_uncertainty_km,
            )
        except ValueError as exc:
            # group_range < ground_distance — geometrically impossible.
            # This means the peak is short-range clutter the mask missed.
            log.warning(
                "[%s] inversion rejected peak at %.0f km (TX-RX %.0f km): %s",
                self.station_id, detection.group_range_km,
                self.ground_distance_km, exc,
            )
            return None

        ts = datetime.now(timezone.utc)
        path = self.writer.write(
            timestamp=ts,
            fix=fix,
            detection=detection,
            radiod_status_dns=self.radiod_status_dns,
            oblique_freq_hz=self.center_freq_hz,
            coherent_seconds=self.coherent_seconds,
            sweep_rate_hz_per_s=self.sweep_rate_hz_per_s,
        )
        log.info(
            "[%s] P=%.0f km h'=%.0f±%.0f km fv=%.3f±%.3f MHz SNR=%.1f dB",
            self.station_id,
            fix.group_range_km, fix.virtual_height_km,
            fix.virtual_height_uncertainty_km,
            fix.equivalent_vertical_freq_mhz,
            fix.equivalent_vertical_freq_uncertainty_mhz,
            detection.snr_db,
        )
        return path

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
            for cpi_samples in self.iq_source:
                if self._stopped.is_set():
                    break
                # Fan out: every transmitter sees the same IQ; each
                # dechirps with its own sweep params, finds its own peak.
                for pipeline in self.pipelines:
                    pipeline.process_cpi(cpi_samples)
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

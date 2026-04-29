"""JSON-Lines writer for codar-sounder ionospheric records.

One file per (radiod_id, station_id, UTC date) triple at::

    /var/lib/codar-sounder/<radiod_id>/<station>/<YYYY>/<MM>/<DD>.jsonl

Each line is a self-contained JSON object: timestamp, group range,
virtual height, equivalent vertical frequency, SNR, plus the geometry
that produced the inversion.  Downstream consumers (HamSCI archive,
TID/space-weather post-processors) parse one line per CPI.

Atomic-ish writes: each record is written with a trailing newline and
``flush()``.  We don't fsync per-record (would dominate I/O cost on
hosts with rotational disks) — at most one record can be lost on crash.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Optional

from codar_sounder.core.invert import IonosphericFix
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
        *,
        radiod_status_dns: str,
        oblique_freq_hz: int,
        coherent_seconds: float,
        sweep_rate_hz_per_s: float,
    ) -> Path:
        """Write one record and return the path it landed in."""
        record: dict[str, Any] = {
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            "client": "codar-sounder",
            "radiod_id": self.radiod_id,
            "radiod_status_dns": radiod_status_dns,
            "station_id": self.station_id,
            "oblique_freq_hz": oblique_freq_hz,
            "sweep_rate_hz_per_s": sweep_rate_hz_per_s,
            "coherent_seconds": coherent_seconds,
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

"""Known CODAR transmitter inventory + audibility ranking.

Loads the vendored transmitter site database (``data/codar-stations.toml``)
and ranks the sites by geodesic distance from the receiver — the CODAR
analogue of superdarn-sounder's ``core/radars.py:audible_radars``.

This is the shared foundation behind two operator surfaces:

  * ``codar-sounder stations [--json]`` — a machine-readable ranked
    inventory (also what sigmond's TUI multi-select consumes); and
  * the ``config init`` whiptail picker, which lets an operator choose
    which transmitters to record instead of hand-writing
    ``[[radiod.transmitter]]`` blocks.

It also owns the **units bridge** from the station DB's human-friendly
``freq_mhz`` to the config/daemon's ``center_freq_hz`` (integer Hz).
Generating ``[[radiod.transmitter]]`` blocks from this canonical table
eliminates the hand-transcription errors that hand-edited configs are
prone to (e.g. a sweep rate off by 1000×).

Geometry comes from the suite-shared ``hamsci_dsp.geometry`` (WGS-84
geodesics via geographiclib).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from hamsci_dsp.geometry import bearing_deg, great_circle_km


# ---------------------------------------------------------------------------
# Band grouping.
#
# A single [[radiod]] block opens ONE wideband IQ channel, so the
# transmitters listed under it must share a band.  These coarse bins
# (drawn from the allocations actually present in codar-stations.toml)
# group the inventory by propagation class so the picker can offer one
# band at a time.  The label is informational; the (lo, hi) edges in MHz
# decide membership.
# ---------------------------------------------------------------------------

BANDS: list[tuple[float, float, str]] = [
    (4.0, 5.2, "4–5 MHz (long-range)"),
    (11.0, 12.9, "12 MHz (mid-range)"),
    (13.0, 14.0, "13 MHz (mid-range)"),
    (23.0, 27.0, "24–26 MHz (short-range)"),
    (40.0, 41.0, "40 MHz (very-short-range)"),
]

# Prime one-hop sky-wave window (km).  Sites inside it are the strongest,
# most reliable targets and the picker pre-checks them by default; sites
# outside it stay listed (multi-hop / skip-zone) but unchecked.  Mirrors
# the 50 km / 2000 km thresholds contract.py:_collect_issues warns on.
PRIME_MIN_KM = 200.0
PRIME_MAX_KM = 2000.0


def band_label(freq_mhz: float) -> str:
    """Return the coarse band label for a centre frequency, or a Hz-rounded
    fallback (``"<n> MHz"``) when the frequency is outside the known bins."""
    for lo, hi, label in BANDS:
        if lo <= freq_mhz < hi:
            return label
    return f"{round(freq_mhz)} MHz"


@dataclass(frozen=True)
class TransmitterCandidate:
    """One CODAR site ranked relative to the receiver."""

    id: str
    freq_mhz: float
    sweep_rate_hz_per_s: float
    sweep_bw_hz: float
    sweep_repetition_hz: float
    tx_lat_deg: float
    tx_lon_deg: float
    association: str
    band: str
    distance_km: float
    bearing_deg: float

    @property
    def center_freq_hz(self) -> int:
        """Integer-Hz centre frequency for the config / daemon."""
        return int(round(self.freq_mhz * 1e6))

    @property
    def in_prime_range(self) -> bool:
        return PRIME_MIN_KM <= self.distance_km <= PRIME_MAX_KM

    def to_tx_block(self) -> dict:
        """Render a ``[[radiod.transmitter]]`` config block.

        This is the units bridge: ``freq_mhz`` → ``center_freq_hz``;
        all other fields carry through in the daemon's units
        (config.REQUIRED_TX_FIELDS).
        """
        return {
            "id": self.id,
            "center_freq_hz": self.center_freq_hz,
            "sweep_rate_hz_per_s": float(self.sweep_rate_hz_per_s),
            "sweep_bw_hz": int(round(self.sweep_bw_hz)),
            "sweep_repetition_hz": float(self.sweep_repetition_hz),
            "tx_lat_deg": float(self.tx_lat_deg),
            "tx_lon_deg": float(self.tx_lon_deg),
        }

    def to_dict(self) -> dict:
        """JSON-friendly view for ``stations --json`` and the TUI."""
        return {
            "id": self.id,
            "freq_mhz": self.freq_mhz,
            "center_freq_hz": self.center_freq_hz,
            "band": self.band,
            "association": self.association,
            "tx_lat_deg": self.tx_lat_deg,
            "tx_lon_deg": self.tx_lon_deg,
            "distance_km": round(self.distance_km, 1),
            "bearing_deg": round(self.bearing_deg, 1),
            "in_prime_range": self.in_prime_range,
            "sweep_rate_hz_per_s": self.sweep_rate_hz_per_s,
            "sweep_bw_hz": self.sweep_bw_hz,
            "sweep_repetition_hz": self.sweep_repetition_hz,
        }


def _find_stations_file() -> Optional[Path]:
    """Locate ``codar-stations.toml`` for editable and packaged installs."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent
        / "data" / "codar-stations.toml",
        Path("/opt/git/sigmond/codar-sounder/data/codar-stations.toml"),
        Path("/usr/local/share/codar-sounder/codar-stations.toml"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_stations(path: Optional[Path] = None) -> dict[str, dict]:
    """Return ``{abbr: station_fields}`` from the station database.

    Raises ``FileNotFoundError`` when the database can't be located.
    """
    if path is None:
        path = _find_stations_file()
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            "codar-stations.toml not found "
            "(looked alongside the repo and in /usr/local/share)"
        )
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return dict(raw.get("stations", {}))


def audible_transmitters(
    receiver_lat: float,
    receiver_lon: float,
    *,
    min_range_km: float = 0.0,
    max_range_km: float = 4000.0,
    bands: Optional[list[str]] = None,
    only: Optional[list[str]] = None,
    stations: Optional[dict[str, dict]] = None,
) -> list[TransmitterCandidate]:
    """Rank known CODAR transmitters by distance from the receiver.

    Returns the candidates within ``[min_range_km, max_range_km]``, sorted
    nearest-first.  ``bands`` (a list of ``band_label`` strings) and
    ``only`` (a list of station abbreviations) further restrict the set
    when non-empty.  ``stations`` overrides the vendored table (for tests).

    Mirrors superdarn-sounder's ``audible_radars`` — distance-only ranking;
    geometry/propagation weighting is a deliberate future increment (see
    superdarn's docs/RADAR-EXPANSION.md Phase B).
    """
    table = stations if stations is not None else load_stations()
    only_set = {a.upper() for a in (only or [])}
    band_set = set(bands or [])

    out: list[TransmitterCandidate] = []
    for abbr, s in table.items():
        if only_set and abbr.upper() not in only_set:
            continue
        try:
            freq_mhz = float(s["freq_mhz"])
            lat = float(s["tx_lat_deg"])
            lon = float(s["tx_lon_deg"])
        except (KeyError, TypeError, ValueError):
            continue

        band = band_label(freq_mhz)
        if band_set and band not in band_set:
            continue

        d = great_circle_km(receiver_lat, receiver_lon, lat, lon)
        if d < min_range_km or d > max_range_km:
            continue

        out.append(TransmitterCandidate(
            id=abbr,
            freq_mhz=freq_mhz,
            sweep_rate_hz_per_s=float(s.get("sweep_rate_hz_per_s", 0.0)),
            sweep_bw_hz=float(s.get("sweep_bw_hz", 0.0)),
            sweep_repetition_hz=float(s.get("sweep_repetition_hz", 1.0)),
            tx_lat_deg=lat,
            tx_lon_deg=lon,
            association=str(s.get("association", "")),
            band=band,
            distance_km=d,
            bearing_deg=bearing_deg(receiver_lat, receiver_lon, lat, lon),
        ))

    out.sort(key=lambda c: c.distance_km)
    return out


def station_to_tx_block(abbr: str, station: dict) -> dict:
    """Units bridge for one raw station-DB entry → a transmitter block.

    Standalone helper for callers that already hold a station dict (and
    don't need a ranked candidate); equivalent to building a
    ``TransmitterCandidate`` and calling ``.to_tx_block()``.
    """
    return {
        "id": abbr,
        "center_freq_hz": int(round(float(station["freq_mhz"]) * 1e6)),
        "sweep_rate_hz_per_s": float(station["sweep_rate_hz_per_s"]),
        "sweep_bw_hz": int(round(float(station["sweep_bw_hz"]))),
        "sweep_repetition_hz": float(station["sweep_repetition_hz"]),
        "tx_lat_deg": float(station["tx_lat_deg"]),
        "tx_lon_deg": float(station["tx_lon_deg"]),
    }

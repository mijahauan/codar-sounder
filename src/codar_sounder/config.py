"""TOML config loader for codar-sounder.

Config layout (per-host, multi-radiod aware):

    [station]            — operator identity + receiver location
    [paths]              — output_dir, log_dir
    [[radiod]]           — one block per local radiod that codar-sounder uses
        id              = "..."
        status_dns      = "..."
        channel_name    = "..."
        [[radiod.transmitter]]
            id, center_freq_hz, sweep_rate_hz_per_s,
            sweep_bw_hz, sweep_repetition_hz,
            tx_lat_deg, tx_lon_deg
    [processing]         — coherent_seconds, range bounds, snr threshold

Daemon is started with --radiod-id <id>; resolve_radiod_block() picks
the matching [[radiod]] block.  Multiple daemons can run on one host,
one per radiod.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path("/etc/codar-sounder/codar-sounder-config.toml")


def load_config(path: Path) -> dict:
    """Load and parse the TOML config; raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def radiod_blocks(config: dict) -> list[dict]:
    """Return the list of [[radiod]] blocks (always a list, even if one)."""
    blocks = config.get("radiod", [])
    if isinstance(blocks, dict):
        return [blocks]
    return list(blocks)


def resolve_radiod_block(config: dict, radiod_id: str | None) -> dict:
    """Pick the [[radiod]] block matching radiod_id.

    If radiod_id is None, returns the first block (single-radiod hosts).
    Raises ValueError if no match.
    """
    blocks = radiod_blocks(config)
    if not blocks:
        raise ValueError("config has no [[radiod]] blocks")
    if radiod_id is None:
        return blocks[0]
    for block in blocks:
        if block.get("id") == radiod_id:
            return block
    have = ", ".join(b.get("id", "<unnamed>") for b in blocks)
    raise ValueError(
        f"no [[radiod]] block with id={radiod_id!r}; have: {have}"
    )


def transmitters(block: dict) -> list[dict]:
    """Return the [[radiod.transmitter]] list for a radiod block."""
    txs = block.get("transmitter", [])
    if isinstance(txs, dict):
        return [txs]
    return list(txs)


def transmitter_freqs(block: dict) -> list[int]:
    """Centre frequencies (Hz) of every transmitter in a radiod block."""
    return sorted({int(t["center_freq_hz"]) for t in transmitters(block)
                   if "center_freq_hz" in t})


REQUIRED_TX_FIELDS = (
    "id",
    "center_freq_hz",
    "sweep_rate_hz_per_s",
    "sweep_bw_hz",
    "sweep_repetition_hz",
    "tx_lat_deg",
    "tx_lon_deg",
)


def missing_tx_fields(tx: dict) -> list[str]:
    """Return required fields absent from a [[radiod.transmitter]] block."""
    return [f for f in REQUIRED_TX_FIELDS if f not in tx]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    import math
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

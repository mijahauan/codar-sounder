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

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path("/etc/codar-sounder/codar-sounder-config.toml")
PER_INSTANCE_CONFIG_DIR = Path("/etc/codar-sounder")


def resolve_config_path(
    instance: Optional[str] = None,
    explicit_path: Optional[Path] = None,
) -> Path:
    """Resolve the config file path per sigmond MULTI-INSTANCE-ARCHITECTURE.md §4.

    Precedence: explicit_path > $CODAR_SOUNDER_CONFIG > per-instance
    /etc/codar-sounder/<instance>.toml (when given and exists) > legacy
    /etc/codar-sounder/codar-sounder-config.toml (with DeprecationWarning
    when --instance was given but the per-instance file is missing).
    """
    if explicit_path is not None:
        return Path(explicit_path)
    env_override = os.environ.get("CODAR_SOUNDER_CONFIG")
    if env_override:
        return Path(env_override)
    if instance:
        per_instance = PER_INSTANCE_CONFIG_DIR / f"{instance}.toml"
        if per_instance.exists():
            return per_instance
        warnings.warn(
            f"per-instance config {per_instance} not found; falling "
            f"back to legacy shared config {DEFAULT_CONFIG_PATH}. "
            f"Migrate this host with `sudo smd instance migrate` "
            f"(MULTI-INSTANCE-ARCHITECTURE.md §6).",
            DeprecationWarning,
            stacklevel=2,
        )
    return DEFAULT_CONFIG_PATH


def extract_reporter_id(config_or_path) -> Optional[str]:
    """Read reporter_id from a per-instance config's [instance] block.

    Accepts either a parsed TOML dict or a Path.  Returns None when no
    [instance] block is present (legacy shared-config world).  Callers
    should NOT fall back to the systemd %i / --instance value as the
    reporter_id during the cutover — %i is typically a radiod identifier,
    not a reporter ID; using it would propagate a misleading value into
    rows.  Leave reporter_id None; row construction falls back to
    radiod_id (the existing legacy `instance` field's semantic).
    """
    if isinstance(config_or_path, dict):
        raw = config_or_path
    else:
        path = Path(config_or_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return None
    inst = raw.get("instance")
    if not isinstance(inst, dict):
        return None
    rid = inst.get("reporter_id")
    if not isinstance(rid, str) or not rid:
        return None
    return rid


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

    Phase 6 cutover (RADIOD-IDENTIFICATION.md §3.1): the mDNS
    multicast status name is the only identifier.  Operators with
    legacy `id`/`status_dns` configs must run
    ``sudo smd radiod migrate --yes``.

    If radiod_id is None, returns the first block (single-radiod hosts).
    Raises ValueError if no match.
    """
    blocks = radiod_blocks(config)
    if not blocks:
        raise ValueError("config has no [[radiod]] blocks")
    if radiod_id is None:
        return blocks[0]
    for block in blocks:
        if block.get("status") == radiod_id:
            return block
    have = ", ".join(b.get("status", "<unnamed>") for b in blocks)
    raise ValueError(
        f"no [[radiod]] block with status={radiod_id!r}; have: {have}"
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

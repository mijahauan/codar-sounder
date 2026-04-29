"""Sigmond client contract v0.5 — inventory and validate JSON builders.

Contract layout (sigmond/docs/CONTRACT-v0.5-DRAFT.md):

  §3   inventory --json — per-instance resource view
  §4   stdout cleanliness
  §11  log level, SIGHUP reload
  §12  validate --json — config validation
  §14  configuration interview — config init/edit
  §15  radiod channel contributions ([[radiod.fragment]])
"""

from __future__ import annotations

import logging
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from codar_sounder.config import (
    haversine_km,
    missing_tx_fields,
    radiod_blocks,
    transmitters,
)
from codar_sounder.version import GIT_INFO


CONTRACT_VERSION = "0.5"


def _client_version() -> str:
    try:
        return pkg_version("codar-sounder")
    except Exception:
        return "0.1.0"


def build_inventory(config: dict, config_path: Path) -> dict:
    """Build the inventory --json payload per contract v0.5 §3.

    One ``instances[]`` entry per configured ``[[radiod.transmitter]]`` —
    a single daemon (one radiod) reports all the CODAR transmitters it
    is dechirping in parallel.  Same shape wspr-recorder uses to report
    multiple receivers under one daemon.
    """
    paths = config.get("paths", {})
    log_dir = paths.get("log_dir", "/var/log/codar-sounder")
    output_dir = paths.get("output_dir", "/var/lib/codar-sounder")

    instances: list[dict] = []
    all_log_paths: dict[str, Any] = {}

    for block in radiod_blocks(config):
        radiod_id = block.get("id", "default")
        status_dns = block.get("status_dns", "")

        for tx in transmitters(block):
            tx_id = tx.get("id", "<unnamed>")
            freq = int(tx.get("center_freq_hz", 0))
            instance: dict[str, Any] = {
                "instance": tx_id,
                "radiod_id": radiod_id,
                "host": "localhost",
                "radiod_status_dns": status_dns,
                "data_destination": None,
                "frequencies_hz": [freq] if freq else [],
                "ka9q_channels": 0,             # we share one channel across TXs
                "required_cores": [],
                "preferred_cores": "worker",
                "disk_writes": [
                    {
                        "path": f"{output_dir}/{radiod_id}/{tx_id}",
                        "mb_per_day": 5,
                        "retention_days": 365,
                    }
                ],
                "uses_timing_calibration": False,
                "provides_timing_calibration": False,
            }
            instances.append(instance)

            all_log_paths[f"{radiod_id}/{tx_id}"] = {
                "process": f"{log_dir}/{radiod_id}.log",
                "products": f"{output_dir}/{radiod_id}/{tx_id}",
            }

    effective_level = logging.getLogger().getEffectiveLevel()
    log_level_name = logging.getLevelName(effective_level)

    payload: dict[str, Any] = {
        "client": "codar-sounder",
        "version": _client_version(),
        "contract_version": CONTRACT_VERSION,
        "config_path": str(config_path),
        "deploy_toml_path": "/opt/git/sigmond/codar-sounder/deploy.toml",
    }

    if GIT_INFO:
        payload["git"] = GIT_INFO

    if all_log_paths:
        payload["log_paths"] = all_log_paths

    payload["log_level"] = log_level_name
    payload["instances"] = instances
    payload["deps"] = {
        "pypi": [
            {"name": "ka9q-python", "version": ">=3.8.0"},
            {"name": "numpy", "version": ">=1.24.0"},
        ],
    }
    payload["issues"] = _collect_issues(config)
    return payload


def build_validate(config: dict, config_path: Path | None = None) -> dict:
    """Build the validate --json payload per contract v0.5 §12."""
    issues = _collect_issues(config)
    payload: dict[str, Any] = {
        "ok": not any(i["severity"] == "fail" for i in issues),
    }
    if config_path is not None:
        payload["config_path"] = str(config_path)
    payload["issues"] = issues
    return payload


def _collect_issues(config: dict) -> list[dict]:
    """Run validation checks and return a list of issue dicts."""
    issues: list[dict] = []

    station = config.get("station", {})
    if not station.get("callsign"):
        issues.append({
            "severity": "warn", "instance": "all",
            "message": "station.callsign is empty",
        })

    rx_lat = station.get("receiver_lat")
    rx_lon = station.get("receiver_lon")
    if rx_lat is None or rx_lon is None:
        issues.append({
            "severity": "fail", "instance": "all",
            "message": "station.receiver_lat / receiver_lon not set "
                       "(needed to compute virtual height + secant law)",
        })

    blocks = radiod_blocks(config)
    if not blocks:
        issues.append({
            "severity": "fail", "instance": "all",
            "message": "no [[radiod]] blocks configured",
        })

    for block in blocks:
        rid = block.get("id", "<unnamed>")
        if not block.get("status_dns"):
            issues.append({
                "severity": "fail", "instance": rid,
                "message": "radiod.status_dns not set",
            })
        if not block.get("channel_name"):
            issues.append({
                "severity": "fail", "instance": rid,
                "message": "radiod.channel_name not set "
                           "(must match the [[radiod.fragment]]-declared channel)",
            })

        txs = transmitters(block)
        if not txs:
            issues.append({
                "severity": "fail", "instance": rid,
                "message": f"radiod {rid!r} has no [[radiod.transmitter]] blocks",
            })

        for tx in txs:
            tx_id = tx.get("id", "<unnamed>")
            missing = missing_tx_fields(tx)
            if missing:
                issues.append({
                    "severity": "fail", "instance": f"{rid}/{tx_id}",
                    "message": f"transmitter missing fields: {', '.join(missing)}",
                })
                continue

            if rx_lat is not None and rx_lon is not None:
                dist = haversine_km(
                    rx_lat, rx_lon,
                    float(tx["tx_lat_deg"]), float(tx["tx_lon_deg"]),
                )
                if dist < 50:
                    issues.append({
                        "severity": "warn", "instance": f"{rid}/{tx_id}",
                        "message": f"TX-RX distance only {dist:.0f} km "
                                   f"— too close for sky-wave propagation",
                    })
                elif dist > 2000:
                    issues.append({
                        "severity": "warn", "instance": f"{rid}/{tx_id}",
                        "message": f"TX-RX distance {dist:.0f} km exceeds 2000 km "
                                   f"— one-hop sky-wave unlikely",
                    })

    return issues

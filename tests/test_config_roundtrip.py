"""Tests for configurator.cmd_config_apply / _serialize_toml — the JSON
round-trip surface the transmitter picker (whiptail + sigmond TUI) writes
through.  The key property: nested [[radiod.transmitter]] arrays-of-tables
serialize to valid TOML and survive a show→apply→show cycle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from codar_sounder import configurator


def _payload():
    return {
        "station": {
            "callsign": "AC0G", "grid_square": "EM38ww",
            "receiver_lat": 38.92, "receiver_lon": -92.29,
        },
        "radiod": [
            {
                "status": "bee1-status.local",
                "channel_name": "codar-4mhz",
                "transmitter": [
                    {"id": "DUCK", "center_freq_hz": 4537180,
                     "sweep_rate_hz_per_s": -25733.913, "sweep_bw_hz": 25734,
                     "sweep_repetition_hz": 1.0, "tx_lat_deg": 36.18035,
                     "tx_lon_deg": -75.750133},
                    {"id": "HATY", "center_freq_hz": 4537180,
                     "sweep_rate_hz_per_s": -25733.913, "sweep_bw_hz": 25734,
                     "sweep_repetition_hz": 1.0, "tx_lat_deg": 35.257217,
                     "tx_lon_deg": -75.519883},
                ],
            },
        ],
    }


def test_serialize_nested_transmitters_is_valid_toml():
    text = configurator._serialize_toml(_payload())
    parsed = tomllib.loads(text)          # must not raise
    txs = parsed["radiod"][0]["transmitter"]
    assert [t["id"] for t in txs] == ["DUCK", "HATY"]
    # Parent-block scalars stay attached to the [[radiod]] header, not
    # captured by the first [[radiod.transmitter]] sub-table.
    assert parsed["radiod"][0]["status"] == "bee1-status.local"
    assert parsed["radiod"][0]["channel_name"] == "codar-4mhz"
    assert parsed["station"]["callsign"] == "AC0G"


def test_apply_then_show_roundtrip(tmp_path: Path):
    cfg = tmp_path / "codar-sounder-config.toml"
    payload = _payload()

    # apply: read JSON from stdin (monkeypatched), write the TOML file.
    import io
    stdin = io.StringIO(json.dumps(payload))
    old_stdin = sys.stdin
    sys.stdin = stdin
    try:
        rc = configurator.cmd_config_apply(SimpleNamespace(config=cfg))
    finally:
        sys.stdin = old_stdin
    assert rc == 0
    assert cfg.is_file()

    # show: the file parses and preserves the transmitter set + units.
    with open(cfg, "rb") as f:
        back = tomllib.load(f)
    txs = back["radiod"][0]["transmitter"]
    assert {t["id"] for t in txs} == {"DUCK", "HATY"}
    assert txs[0]["center_freq_hz"] == 4537180
    assert txs[0]["sweep_rate_hz_per_s"] == pytest.approx(-25733.913)


def test_apply_rejects_unknown_section(tmp_path: Path):
    import io
    sys.stdin, old = io.StringIO(json.dumps({"bogus": {}})), sys.stdin
    try:
        rc = configurator.cmd_config_apply(
            SimpleNamespace(config=tmp_path / "c.toml"))
    finally:
        sys.stdin = old
    assert rc == 2

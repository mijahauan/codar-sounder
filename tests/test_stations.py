"""Tests for core/stations.py — the CODAR transmitter inventory + picker
foundation (audibility ranking, the MHz→Hz units bridge, band binning),
plus a regression guard that the shipped template's example transmitter
blocks agree with the canonical station database."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from codar_sounder.core.stations import (
    BANDS,
    PRIME_MAX_KM,
    PRIME_MIN_KM,
    TransmitterCandidate,
    audible_transmitters,
    band_label,
    load_stations,
    station_to_tx_block,
)

_REPO = Path(__file__).resolve().parent.parent

# A tiny synthetic table so ranking tests don't depend on the vendored DB.
_FAKE = {
    "NEAR": {  # ~150 km E of the receiver (skip-zone-ish, < prime)
        "freq_mhz": 4.5, "sweep_rate_hz_per_s": -25733.913,
        "sweep_bw_hz": 25734, "sweep_repetition_hz": 1,
        "tx_lat_deg": 38.92, "tx_lon_deg": -90.5, "association": "X",
    },
    "MID": {   # ~1000 km, prime one-hop
        "freq_mhz": 13.45, "sweep_rate_hz_per_s": -49629.688,
        "sweep_bw_hz": 49630, "sweep_repetition_hz": 1,
        "tx_lat_deg": 38.92, "tx_lon_deg": -80.8, "association": "Y",
    },
    "FAR": {   # ~3500 km, beyond one-hop
        "freq_mhz": 25.4, "sweep_rate_hz_per_s": -101097.519,
        "sweep_bw_hz": 202195, "sweep_repetition_hz": 0.5,
        "tx_lat_deg": 38.92, "tx_lon_deg": -52.0, "association": "Z",
    },
}
_RX = (38.92, -92.29)


def test_load_stations_real_db():
    st = load_stations()
    assert len(st) > 40                      # ~60 sites vendored
    duck = st["DUCK"]
    assert duck["freq_mhz"] == pytest.approx(4.53718)
    assert duck["sweep_rate_hz_per_s"] == pytest.approx(-25733.913)


def test_audible_sorted_nearest_first():
    cands = audible_transmitters(*_RX, stations=_FAKE)
    ids = [c.id for c in cands]
    assert ids == ["NEAR", "MID", "FAR"]
    assert cands[0].distance_km < cands[1].distance_km < cands[2].distance_km


def test_range_filter():
    cands = audible_transmitters(*_RX, min_range_km=300, max_range_km=2000,
                                 stations=_FAKE)
    assert [c.id for c in cands] == ["MID"]   # NEAR too close, FAR too far


def test_only_and_band_filters():
    only = audible_transmitters(*_RX, only=["mid"], stations=_FAKE)  # case-insensitive
    assert [c.id for c in only] == ["MID"]
    by_band = audible_transmitters(*_RX, bands=["13 MHz (mid-range)"],
                                   stations=_FAKE)
    assert [c.id for c in by_band] == ["MID"]


def test_prime_range_flag():
    cands = {c.id: c for c in audible_transmitters(*_RX, stations=_FAKE)}
    assert cands["NEAR"].in_prime_range is False    # < PRIME_MIN_KM
    assert cands["MID"].in_prime_range is True
    assert cands["FAR"].in_prime_range is False     # > PRIME_MAX_KM
    assert PRIME_MIN_KM < PRIME_MAX_KM


def test_units_bridge_mhz_to_hz():
    cand = audible_transmitters(*_RX, only=["MID"], stations=_FAKE)[0]
    block = cand.to_tx_block()
    assert block["center_freq_hz"] == 13450000      # 13.45 MHz → int Hz
    assert isinstance(block["center_freq_hz"], int)
    assert block["sweep_rate_hz_per_s"] == pytest.approx(-49629.688)
    assert block["id"] == "MID"
    # Required-field parity with the daemon/contract schema.
    from codar_sounder.config import missing_tx_fields
    assert missing_tx_fields(block) == []


def test_station_to_tx_block_matches_candidate():
    direct = station_to_tx_block("MID", _FAKE["MID"])
    via_cand = audible_transmitters(*_RX, only=["MID"],
                                    stations=_FAKE)[0].to_tx_block()
    assert direct == via_cand


def test_fractional_sweep_repetition_preserved():
    block = station_to_tx_block("FAR", _FAKE["FAR"])
    assert block["sweep_repetition_hz"] == pytest.approx(0.5)


def test_band_label_bins():
    assert band_label(4.537) == "4–5 MHz (long-range)"
    assert band_label(13.45) == "13 MHz (mid-range)"
    assert band_label(25.4) == "24–26 MHz (short-range)"
    assert band_label(40.75) == "40 MHz (very-short-range)"
    # Outside every bin → Hz-rounded fallback, never a crash.
    assert band_label(7.0).endswith("MHz")


def test_every_vendored_station_lands_in_a_band():
    """No real station should hit the fallback bin — keeps BANDS honest."""
    known = {label for _, _, label in BANDS}
    for abbr, s in load_stations().items():
        assert band_label(float(s["freq_mhz"])) in known, abbr


def test_template_examples_match_canonical_db():
    """The shipped template's example [[radiod.transmitter]] blocks must
    agree with data/codar-stations.toml — the regression guard for the
    historical 1000× sweep-rate transcription bug."""
    db = load_stations()
    template = _REPO / "config" / "codar-sounder-config.toml.template"
    text = template.read_text()
    # Parse only the active (uncommented) transmitter blocks.
    cfg = tomllib.loads(text)
    blocks = cfg.get("radiod", [])
    if isinstance(blocks, dict):
        blocks = [blocks]
    seen = 0
    for rb in blocks:
        txs = rb.get("transmitter", [])
        if isinstance(txs, dict):
            txs = [txs]
        for tx in txs:
            expected = station_to_tx_block(tx["id"], db[tx["id"]])
            assert tx["center_freq_hz"] == expected["center_freq_hz"], tx["id"]
            assert tx["sweep_rate_hz_per_s"] == pytest.approx(
                expected["sweep_rate_hz_per_s"]), tx["id"]
            assert tx["sweep_bw_hz"] == expected["sweep_bw_hz"], tx["id"]
            seen += 1
    assert seen >= 1

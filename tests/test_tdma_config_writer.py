"""Tests for tdma-scan --write-config in-place TOML rewriter.

Verifies:
  * existing tdma_offset_samples lines get their value replaced
  * new lines get inserted after the matching `id = ...`
  * comments and unrelated whitespace are preserved
  * scope is honoured (only the requested radiod block is touched)
  * unknown radiod_id raises ValueError without writing
"""
from __future__ import annotations

import sys
from pathlib import Path
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from codar_sounder.tdma_config_writer import (
    _rewrite,
    update_tdma_offsets_in_toml,
)


def _config(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


SIMPLE_CONFIG = _config("""
    [station]
    callsign = "AC0G"
    receiver_lat = 36.71
    receiver_lon = -75.93

    [[radiod]]
    status = "bee1-status.local"
    channel_name = "codar-4mhz"
    sample_rate_hz = 64000

    # The east-coast 4.5 MHz CODAR group.
    [[radiod.transmitter]]
    id = "DUCK"
    center_freq_hz = 4537180
    sweep_rate_hz_per_s = -25733.913
    sweep_bw_hz = 25734
    tx_lat_deg = 36.18
    tx_lon_deg = -75.74

    [[radiod.transmitter]]
    id = "LISL"
    center_freq_hz = 4537180
    sweep_rate_hz_per_s = -25733.913
    sweep_bw_hz = 25734
    tx_lat_deg = 36.69
    tx_lon_deg = -75.92
    tdma_offset_samples = 0          # placeholder, will be overwritten

    [processing]
    coherent_seconds = 60
    snr_threshold_db = 5.0
""")


class TestRewrite:

    def test_replaces_existing_tdma_line(self):
        out_lines, n_changed, n_inserted = _rewrite(
            SIMPLE_CONFIG,
            radiod_id="bee1-status.local",
            offsets={"LISL": 21385},
        )
        assert n_changed == 1
        assert n_inserted == 0
        result = "\n".join(out_lines)
        assert "tdma_offset_samples = 21385" in result
        assert "tdma_offset_samples = 0" not in result

    def test_inserts_new_line_after_id(self):
        out_lines, n_changed, n_inserted = _rewrite(
            SIMPLE_CONFIG,
            radiod_id="bee1-status.local",
            offsets={"DUCK": 175},
        )
        assert n_changed == 0
        assert n_inserted == 1
        result = "\n".join(out_lines)
        # The insert sits immediately after the DUCK id line.
        idx_id = result.index('id = "DUCK"')
        idx_off = result.index('tdma_offset_samples = 175')
        assert idx_id < idx_off
        # No insertion happened to LISL (we didn't request its offset).
        assert result.count("tdma_offset_samples =") == 2  # original (0) + new (175)
        assert "tdma_offset_samples = 0" in result

    def test_handles_replace_and_insert_in_one_pass(self):
        out_lines, n_changed, n_inserted = _rewrite(
            SIMPLE_CONFIG,
            radiod_id="bee1-status.local",
            offsets={"DUCK": 175, "LISL": 21385},
        )
        assert n_changed == 1     # LISL
        assert n_inserted == 1    # DUCK
        result = "\n".join(out_lines)
        assert "tdma_offset_samples = 175" in result
        assert "tdma_offset_samples = 21385" in result

    def test_preserves_comments_and_spacing(self):
        out_lines, *_ = _rewrite(
            SIMPLE_CONFIG,
            radiod_id="bee1-status.local",
            offsets={"DUCK": 100},
        )
        result = "\n".join(out_lines)
        # The comment line above the first transmitter block survives.
        assert "# The east-coast 4.5 MHz CODAR group." in result
        # The trailing comment on the LISL placeholder line survives
        # (we didn't touch LISL in this test).
        assert "# placeholder, will be overwritten" in result

    def test_other_radiod_blocks_untouched(self):
        # Two [[radiod]] blocks; only the named one should be edited.
        cfg = _config("""
            [[radiod]]
            status = "rx-A.local"
            channel_name = "ca"
            [[radiod.transmitter]]
            id = "T1"
            center_freq_hz = 1000

            [[radiod]]
            status = "rx-B.local"
            channel_name = "cb"
            [[radiod.transmitter]]
            id = "T1"
            center_freq_hz = 2000
        """)
        out_lines, n_changed, n_inserted = _rewrite(
            cfg, radiod_id="rx-B.local", offsets={"T1": 999},
        )
        assert n_inserted == 1
        result = "\n".join(out_lines)
        # The T1 in rx-A was NOT touched.
        before_rxb = result[: result.index("rx-B.local")]
        assert "tdma_offset_samples" not in before_rxb
        # The T1 in rx-B WAS touched.
        after_rxb = result[result.index("rx-B.local") :]
        assert "tdma_offset_samples = 999" in after_rxb

    def test_unknown_radiod_id_raises(self):
        with pytest.raises(ValueError, match="no \\[\\[radiod\\]\\] block"):
            _rewrite(
                SIMPLE_CONFIG,
                radiod_id="does-not-exist",
                offsets={"DUCK": 1},
            )


class TestAtomicWrite:

    def test_atomic_write_creates_intended_content(self, tmp_path):
        cfg = tmp_path / "codar.toml"
        cfg.write_text(SIMPLE_CONFIG)
        n_changed, n_inserted = update_tdma_offsets_in_toml(
            cfg, radiod_id="bee1-status.local",
            offsets={"DUCK": 175, "LISL": 21385},
        )
        assert (n_changed, n_inserted) == (1, 1)
        new_text = cfg.read_text()
        assert "tdma_offset_samples = 175" in new_text
        assert "tdma_offset_samples = 21385" in new_text

    def test_no_match_leaves_file_untouched(self, tmp_path):
        cfg = tmp_path / "codar.toml"
        cfg.write_text(SIMPLE_CONFIG)
        before = cfg.read_text()
        with pytest.raises(ValueError):
            update_tdma_offsets_in_toml(
                cfg, radiod_id="bee1-status.local",
                offsets={"NONEXISTENT": 999},
            )
        assert cfg.read_text() == before

    def test_unknown_radiod_leaves_file_untouched(self, tmp_path):
        cfg = tmp_path / "codar.toml"
        cfg.write_text(SIMPLE_CONFIG)
        before = cfg.read_text()
        with pytest.raises(ValueError):
            update_tdma_offsets_in_toml(
                cfg, radiod_id="rx-Z",
                offsets={"DUCK": 1},
            )
        assert cfg.read_text() == before

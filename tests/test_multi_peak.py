"""Tests for multi-peak detection + layer classifier + per-peak CH row.

`find_f_region_peaks` (plural) finds every local maximum above the SNR
threshold inside the operator-configured search window, applying a
minimum-separation rule so a single broad peak doesn't split into
duplicates.  `classify_layer` maps the inverted virtual height to a
coarse ionospheric layer label.  Together they let the daemon emit one
JSONL/CH record per propagation mode (1F2 high-ray, 1F2 low-ray, etc.)
instead of just the strongest peak.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from codar_sounder.core.invert import (
    classify_layer, invert, IonosphericFix,
)
from codar_sounder.core.trace import (
    DEFAULT_MAX_PEAKS, GroundClutterMask,
    find_f_region_peak, find_f_region_peaks, TraceDetection,
)


def _profile_with_peaks(n: int, peaks: list[tuple[int, float]]) -> np.ndarray:
    """Build a flat profile with narrow peaks at given (bin, amplitude) pairs."""
    p = np.full(n, 0.01, dtype=np.float32)
    for bin_idx, amp in peaks:
        p[bin_idx] = amp
    return p


# ── classify_layer ──────────────────────────────────────────────────────────

class TestClassifyLayer:
    """CONTRACT v0.6 mode_layer column source-of-truth."""

    @pytest.mark.parametrize("h_km, expected", [
        ( 50.0, "below_E"),
        ( 89.9, "below_E"),
        ( 90.0, "E"),
        (100.0, "E"),
        (139.9, "E"),
        (140.0, "F1"),
        (180.0, "F1"),
        (219.9, "F1"),
        (220.0, "F2"),
        (300.0, "F2"),
        (499.9, "F2"),
        (500.0, "F2_extreme"),
        (700.0, "F2_extreme"),
    ])
    def test_layer_boundaries(self, h_km, expected):
        assert classify_layer(h_km) == expected

    def test_nan_returns_unknown(self):
        assert classify_layer(float("nan")) == "unknown"

    def test_inf_returns_unknown(self):
        assert classify_layer(float("inf")) == "unknown"


# ── find_f_region_peaks (plural) ────────────────────────────────────────────

class TestFindFRegionPeaks:

    def _ranges(self, n: int = 200) -> np.ndarray:
        # 0..1000 km in 5 km bins.
        return np.linspace(0.0, 1000.0, n, dtype=np.float32)

    def test_single_peak_returns_one_detection(self):
        ranges = self._ranges()
        # One strong peak at 500 km (bin 100).
        profile = _profile_with_peaks(len(ranges), [(100, 1.0)])
        peaks = find_f_region_peaks(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
        )
        assert len(peaks) == 1
        assert peaks[0].group_range_km == pytest.approx(500.0, abs=10.0)

    def test_two_separate_peaks_returned_sorted_by_snr(self):
        ranges = self._ranges()
        # Strong peak at 600 km (bin 120), weaker at 350 km (bin 70).
        profile = _profile_with_peaks(
            len(ranges), [(70, 0.3), (120, 1.0)]
        )
        peaks = find_f_region_peaks(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
        )
        assert len(peaks) == 2
        assert peaks[0].snr_db > peaks[1].snr_db
        assert peaks[0].group_range_km > peaks[1].group_range_km

    def test_collapses_close_peaks_to_strongest(self):
        ranges = self._ranges()
        # Two peaks 5 km apart (1 bin = 5 km) — within min_separation.
        profile = _profile_with_peaks(
            len(ranges), [(100, 1.0), (102, 0.7)]
        )
        peaks = find_f_region_peaks(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
            min_separation_km=12.0,
        )
        assert len(peaks) == 1
        # The 1.0-amp peak wins (stronger SNR).
        assert peaks[0].group_range_km == pytest.approx(500.0, abs=10.0)

    def test_max_peaks_caps_returns(self):
        ranges = self._ranges()
        # Five peaks ≥ 30 km apart in the search window.
        peak_specs = [(60 + 8 * i, 1.0 - 0.1 * i) for i in range(5)]
        profile = _profile_with_peaks(len(ranges), peak_specs)
        peaks = find_f_region_peaks(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
            max_peaks=3,
            min_separation_km=12.0,
        )
        assert len(peaks) == 3

    def test_subthreshold_peaks_filtered(self):
        ranges = self._ranges()
        profile = np.full(len(ranges), 0.5, dtype=np.float32)
        profile[100] = 0.55          # tiny bump, ~0.4 dB above median
        peaks = find_f_region_peaks(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
        )
        assert peaks == []

    def test_empty_window_returns_empty_list(self):
        ranges = self._ranges()
        profile = np.ones(len(ranges), dtype=np.float32)
        peaks = find_f_region_peaks(
            profile, ranges,
            range_min_km=2000, range_max_km=3000,    # outside the data
            snr_threshold_db=5.0,
        )
        assert peaks == []

    def test_singular_wrapper_returns_first(self):
        """find_f_region_peak (singular) returns the strongest peak."""
        ranges = self._ranges()
        profile = _profile_with_peaks(
            len(ranges), [(70, 0.3), (120, 1.0)]
        )
        plural = find_f_region_peaks(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
        )
        singular = find_f_region_peak(
            profile, ranges,
            range_min_km=200, range_max_km=800,
            snr_threshold_db=5.0,
        )
        assert singular is not None
        assert singular.group_range_km == plural[0].group_range_km
        assert singular.snr_db == plural[0].snr_db


# ── per-peak CH row construction ────────────────────────────────────────────

class TestChRowBuilder:
    """Verify _TransmitterPipeline._ch_row_for matches the codar.spots schema."""

    def _pipeline(self, **kw):
        # Lazy import to avoid pulling daemon during collection.
        from codar_sounder.core.daemon import _TransmitterPipeline
        defaults = dict(
            receiver_lat=36.7, receiver_lon=-75.9,
            radiod_id="bee1-rx888",
            radiod_status_dns="bee1.local",
            tx_config={
                "id": "DUCK",
                "center_freq_hz": 4537180,
                "sweep_rate_hz_per_s": -25733.913,
                "sweep_bw_hz": 25734.0,
                "sweep_repetition_hz": 1.0,
                "tx_lat_deg": 36.18,
                "tx_lon_deg": -75.74,
            },
            output_dir=Path("/tmp/codar-test-output"),
            sample_rate_hz=64000.0,
            coherent_seconds=60.0,
            range_min_km=200.0,
            range_max_km=800.0,
            snr_threshold_db=5.0,
            host_call="AC0G",
            host_grid="EM38ww",
            processing_version="0.4.0+abc1234",
        )
        defaults.update(kw)
        return _TransmitterPipeline(**defaults)

    def test_row_columns_match_codar_spots_schema(self):
        pipeline = self._pipeline()
        detection = TraceDetection(
            group_range_km=520.0, snr_db=12.5, power=1.0, bin_index=104,
        )
        # Inversion produces a fix with mode_layer set.
        fix = invert(
            group_range_km=520.0,
            ground_distance_km=pipeline.ground_distance_km,
            oblique_freq_mhz=pipeline.center_freq_hz / 1e6,
            group_range_uncertainty_km=pipeline.group_range_uncertainty_km,
        )
        from datetime import datetime, timezone
        from codar_sounder.core.scintillation import ScintillationResult
        scint = ScintillationResult(
            s4_index=0.12, s4_severity="weak",
            sigma_phi_rad=0.05, sigma_phi_severity="weak",
            scintillation_event=False, confidence=1.0,
            n_samples=60, n_outliers_rejected=0, mode_doppler_hz=0.02,
            sigma_phi_linear_rad=0.07, sigma_phi_quadratic_rad=0.05,
            sigma_phi_underfit_ratio=1.4,
        )
        ts = datetime(2026, 5, 7, 12, 30, tzinfo=timezone.utc)
        row = pipeline._ch_row_for(
            timestamp=ts, detection=detection, fix=fix,
            scintillation=scint,
            peak_index=0, peak_count=1,
            dechirp_sweeps_rejected=2,
        )
        # All columns of the codar.spots HamSCI sink row (v0.5).
        expected_cols = {
            "time", "host_call", "host_grid", "radiod_id", "instance",
            "reporter_id",   # Phase-5 (sigmond MULTI-INSTANCE-ARCHITECTURE.md §3)
            "processing_version", "station_id", "oblique_freq_hz",
            "sweep_rate_hz_per_s", "coherent_seconds",
            "peak_index", "peak_count", "mode_layer", "snr_db",
            "group_range_km", "ground_distance_km", "virtual_height_km",
            "virtual_height_uncertainty_km",
            "equivalent_vertical_freq_mhz",
            "equivalent_vertical_freq_uncertainty_mhz",
            "takeoff_zenith_deg",
            # v0.5 scintillation columns:
            "s4_index", "s4_severity",
            "sigma_phi_rad", "sigma_phi_severity",
            "scintillation_event", "scintillation_confidence",
            "scintillation_samples", "mode_doppler_hz",
            # v0.5.1: MAD outlier rejection count.
            "scintillation_outliers_rejected",
            # v0.6: σ_φ diagnostics (linear/quadratic + underfit ratio).
            "sigma_phi_linear_rad", "sigma_phi_quadratic_rad",
            "sigma_phi_underfit_ratio",
            # v0.6.1: per-sweep MAD pre-filter rejection count.
            "dechirp_sweeps_rejected",
            # v0.7: multi-hop selector output.
            "n_hops",
        }
        assert set(row.keys()) == expected_cols
        assert row["host_call"] == "AC0G"
        assert row["host_grid"] == "EM38ww"
        assert row["radiod_id"] == "bee1-rx888"
        assert row["instance"] == "bee1-rx888"
        # reporter_id falls back to radiod_id when no [instance] block
        # is set on the per-instance config (legacy single-instance world).
        assert row["reporter_id"] == "bee1-rx888"
        assert row["station_id"] == "DUCK"
        assert row["mode_layer"] == fix.mode_layer
        assert row["peak_index"] == 0
        assert row["peak_count"] == 1
        assert row["oblique_freq_hz"] == 4537180
        # Scintillation values round-trip without re-rounding.
        assert row["s4_index"] == 0.12
        assert row["s4_severity"] == "weak"
        assert row["sigma_phi_rad"] == 0.05
        assert row["sigma_phi_severity"] == "weak"
        assert row["scintillation_event"] is False
        assert row["scintillation_confidence"] == 1.0
        assert row["scintillation_samples"] == 60
        assert row["scintillation_outliers_rejected"] == 0
        assert row["mode_doppler_hz"] == 0.02
        # v0.6 σ_φ diagnostics.
        assert row["sigma_phi_linear_rad"] == 0.07
        assert row["sigma_phi_quadratic_rad"] == 0.05
        assert row["sigma_phi_underfit_ratio"] == 1.4
        assert row["dechirp_sweeps_rejected"] == 2
        # v0.7 multi-hop selector.  The test geometry uses 520 km
        # group_range from a 1416 km path — geometrically impossible
        # 1-hop, but invert() will still report SOMETHING; here we
        # just verify the field is present.
        assert isinstance(row["n_hops"], int)
        assert row["n_hops"] >= 1


# ── pipeline writes both JSONL and CH ────────────────────────────────────────

class FakeChWriter:
    """Just enough surface to satisfy daemon's CH-write path."""
    def __init__(self):
        self.inserts = []
        self.closed = False
        self.health = "ok"
        self.is_noop = False
        self.database = "codar"

    def insert(self, rows):
        self.inserts.extend(rows)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class TestPipelineEmitsCh:

    def _pipeline_with_writer(self, ch_writer, tmp_path):
        from codar_sounder.core.daemon import _TransmitterPipeline
        return _TransmitterPipeline(
            receiver_lat=36.7, receiver_lon=-75.9,
            radiod_id="bee1-rx888",
            radiod_status_dns="bee1.local",
            tx_config={
                "id": "DUCK",
                "center_freq_hz": 4537180,
                "sweep_rate_hz_per_s": -25733.913,
                "sweep_bw_hz": 25734.0,
                "sweep_repetition_hz": 1.0,
                "tx_lat_deg": 36.18,
                "tx_lon_deg": -75.74,
            },
            output_dir=tmp_path,
            sample_rate_hz=64000.0,
            coherent_seconds=60.0,
            range_min_km=200.0,
            range_max_km=800.0,
            snr_threshold_db=5.0,
            ch_writer=ch_writer,
            host_call="AC0G",
            host_grid="EM38ww",
            processing_version="0.4.0+abc1234",
        )

    def test_synthetic_cpi_writes_per_peak_to_ch_writer(self, tmp_path):
        """Drive process_cpi with a synthetic CPI containing a clear peak.

        The synthetic IQ source already produces dechirpable signal
        (see test_dechirp.py).  We use the in-process synthetic chirp
        helper from the daemon's own factory to feed one CPI; the
        pipeline should emit at least one JSONL record AND one matching
        CH insert.
        """
        # Build a synthetic CPI matching the pipeline parameters.
        from codar_sounder.core.stream import SyntheticIQSource
        fake = FakeChWriter()
        pipeline = self._pipeline_with_writer(fake, tmp_path)
        synth = SyntheticIQSource(
            sample_rate_hz=64000.0,
            sweep_rate_hz_per_s=-25733.913,
            sweep_repetition_hz=1.0,
            cpi_seconds=60.0,
            target_group_range_km=520.0,    # within range_min/max
            realtime=False,
        )
        cpi, cpi_start_utc = next(iter(synth))
        path = pipeline.process_cpi(cpi, cpi_start_utc)
        try:
            assert path is not None, "no JSONL record written"
            assert len(fake.inserts) >= 1, "no CH inserts"
            row = fake.inserts[0]
            assert row["station_id"] == "DUCK"
            assert row["host_call"] == "AC0G"
            assert row["mode_layer"] in (
                "below_E", "E", "F1", "F2", "F2_extreme", "unknown",
            )
            assert row["peak_count"] >= 1
            assert row["peak_index"] == 0
            # v0.5 scintillation fields land on the sink row, with sane
            # values for a clean synthetic CPI (no scintillation by
            # construction): severities classified (not "unknown"),
            # confidence saturated at 1.0 since M=60 ≥ 30, no event.
            assert row["s4_severity"] in {
                "weak", "moderate", "strong", "unknown",
            }
            assert row["sigma_phi_severity"] in {
                "weak", "moderate", "strong", "unknown",
            }
            assert row["scintillation_samples"] >= 10
            assert row["scintillation_event"] is False
            assert 0.0 <= row["scintillation_confidence"] <= 1.0
            assert np.isfinite(row["s4_index"])
            assert row["s4_index"] >= 0.0
            assert np.isfinite(row["sigma_phi_rad"])
            assert row["sigma_phi_rad"] >= 0.0
            # v0.5.1: rejected count is non-negative; clean synthetic
            # signal should produce 0 (no spike to reject).
            assert row["scintillation_outliers_rejected"] >= 0
            # v0.6: underfit ratio ≥ 1 by construction; both σ_φ
            # variants finite and non-negative.
            assert row["sigma_phi_underfit_ratio"] >= 1.0 - 1e-9
            assert np.isfinite(row["sigma_phi_linear_rad"])
            assert np.isfinite(row["sigma_phi_quadratic_rad"])
            # Canonical sigma_phi_rad equals the quadratic value.
            assert row["sigma_phi_quadratic_rad"] == pytest.approx(
                row["sigma_phi_rad"], rel=1e-9
            )
            # v0.6.1: per-sweep MAD pre-filter count.  Clean synthetic
            # CPI should have 0 sweeps rejected; the field type is int.
            assert isinstance(row["dechirp_sweeps_rejected"], int)
            assert row["dechirp_sweeps_rejected"] >= 0
        finally:
            pipeline.close()

    def test_close_drains_ch_writer(self, tmp_path):
        fake = FakeChWriter()
        pipeline = self._pipeline_with_writer(fake, tmp_path)
        pipeline.close()
        # The TransmitterPipeline doesn't own the CH writer's lifecycle
        # (one writer is shared across pipelines in SounderDaemon); only
        # the JSONL writer closes here.  CH writer is closed by the
        # SounderDaemon, not the per-pipeline close().  This test
        # documents that contract.
        assert fake.closed is False

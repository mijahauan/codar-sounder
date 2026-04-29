"""Tests for core/trace.py — peak extraction with ground-clutter masking."""

from __future__ import annotations

import numpy as np
import pytest

from codar_sounder.core.trace import (
    GroundClutterMask,
    TraceDetection,
    find_f_region_peak,
)


def _profile_with_peak(
    n_bins: int,
    peak_bin: int,
    peak_power: float,
    noise_floor: float = 1.0,
) -> np.ndarray:
    """Build a synthetic range profile with one peak above a noise floor."""
    profile = np.full(n_bins, noise_floor, dtype=np.float32)
    # Spread the peak across 3 bins (mimics finite range resolution).
    for offset, weight in [(-1, 0.5), (0, 1.0), (1, 0.5)]:
        idx = peak_bin + offset
        if 0 <= idx < n_bins:
            profile[idx] += peak_power * weight
    return profile


# ---------------------------------------------------------------------------
# GroundClutterMask
# ---------------------------------------------------------------------------

class TestGroundClutterMask:

    def test_empty_estimate_is_zero(self):
        mask = GroundClutterMask(window=10)
        est = mask.estimate(50)
        assert est.shape == (50,)
        assert np.all(est == 0)

    def test_after_one_profile_estimate_equals_input(self):
        mask = GroundClutterMask(window=10)
        profile = np.linspace(0, 10, 50, dtype=np.float32)
        mask.update(profile)
        est = mask.estimate(50)
        assert np.allclose(est, profile)

    def test_median_filters_transient(self):
        """A burst in one of N profiles should NOT raise the median there."""
        mask = GroundClutterMask(window=5)
        baseline = np.ones(20, dtype=np.float32)
        for _ in range(4):
            mask.update(baseline)
        # Inject a transient at bin 10
        burst = baseline.copy()
        burst[10] = 100.0
        mask.update(burst)
        est = mask.estimate(20)
        # Median of [1,1,1,1,100] at bin 10 is 1, not affected by the burst.
        assert est[10] == pytest.approx(1.0, abs=0.01)

    def test_window_size_caps_buffer(self):
        mask = GroundClutterMask(window=3)
        for v in range(10):
            mask.update(np.full(5, float(v), dtype=np.float32))
        # Only the last 3 profiles remain → median of values 7,8,9 is 8.
        est = mask.estimate(5)
        assert np.all(est == 8.0)
        assert mask.n_observations == 3

    def test_subtract_clamps_negative_to_zero(self):
        mask = GroundClutterMask(window=5)
        # Build clutter at level 5
        for _ in range(5):
            mask.update(np.full(10, 5.0, dtype=np.float32))
        # New profile has a value below clutter at bin 0
        new = np.full(10, 5.0, dtype=np.float32)
        new[0] = 0.0
        residual = mask.subtract(new)
        assert residual[0] == 0.0
        assert np.all(residual >= 0)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            GroundClutterMask(window=0)


# ---------------------------------------------------------------------------
# find_f_region_peak
# ---------------------------------------------------------------------------

class TestFindFRegionPeak:

    def _setup(self, n_bins: int = 100):
        # 10 km per bin, ranges 0..1000 km
        ranges = np.arange(n_bins, dtype=np.float32) * 10.0
        return ranges

    def test_finds_strong_peak_in_window(self):
        ranges = self._setup()
        # peak at bin 50 → range 500 km, large amplitude
        profile = _profile_with_peak(100, peak_bin=50, peak_power=100.0)
        det = find_f_region_peak(
            profile, ranges,
            range_min_km=200.0, range_max_km=800.0,
            snr_threshold_db=5.0,
        )
        assert det is not None
        assert det.group_range_km == pytest.approx(500.0)
        assert det.snr_db > 5.0
        assert det.bin_index == 50

    def test_peak_outside_window_is_ignored(self):
        ranges = self._setup()
        # Big peak at bin 10 (100 km) — outside the F-region window.
        profile = _profile_with_peak(100, peak_bin=10, peak_power=100.0)
        det = find_f_region_peak(
            profile, ranges,
            range_min_km=200.0, range_max_km=800.0,
            snr_threshold_db=5.0,
        )
        # The remainder of the profile is uniform noise; SNR fails.
        assert det is None

    def test_subthreshold_peak_returns_none(self):
        ranges = self._setup()
        # Tiny peak; SNR won't meet 20 dB threshold.
        profile = _profile_with_peak(100, peak_bin=50, peak_power=0.5)
        det = find_f_region_peak(
            profile, ranges,
            range_min_km=200.0, range_max_km=800.0,
            snr_threshold_db=20.0,
        )
        assert det is None

    def test_clutter_mask_removes_persistent_clutter(self):
        ranges = self._setup()
        # Build up a stationary "clutter" profile with a peak at bin 30.
        clutter_profile = _profile_with_peak(100, peak_bin=30, peak_power=50.0)
        mask = GroundClutterMask(window=10)
        for _ in range(10):
            mask.update(clutter_profile)

        # Now a new profile has the SAME clutter peak plus a transient
        # F-region peak at bin 60.
        new_profile = clutter_profile.copy()
        new_profile += _profile_with_peak(100, peak_bin=60, peak_power=20.0)

        det = find_f_region_peak(
            new_profile, ranges,
            range_min_km=200.0, range_max_km=800.0,
            snr_threshold_db=5.0,
            clutter_mask=mask,
        )
        # Without the clutter mask, the bin-30 clutter would dominate
        # (it's outside the window — so it'd actually be ignored), but
        # the test is the F-region peak survives the masking.
        assert det is not None
        assert det.bin_index == 60
        assert det.group_range_km == pytest.approx(600.0)

    def test_returns_trace_detection(self):
        ranges = self._setup()
        profile = _profile_with_peak(100, peak_bin=50, peak_power=100.0)
        det = find_f_region_peak(
            profile, ranges,
            range_min_km=200.0, range_max_km=800.0,
            snr_threshold_db=5.0,
        )
        assert isinstance(det, TraceDetection)

    def test_invalid_range_window(self):
        ranges = self._setup()
        profile = _profile_with_peak(100, peak_bin=50, peak_power=100.0)
        with pytest.raises(ValueError):
            find_f_region_peak(
                profile, ranges,
                range_min_km=800.0, range_max_km=200.0,    # inverted!
                snr_threshold_db=5.0,
            )

    def test_shape_mismatch_rejected(self):
        ranges = self._setup(100)
        profile = np.zeros(50, dtype=np.float32)        # wrong length
        with pytest.raises(ValueError, match="shape"):
            find_f_region_peak(
                profile, ranges,
                range_min_km=200.0, range_max_km=800.0,
                snr_threshold_db=5.0,
            )

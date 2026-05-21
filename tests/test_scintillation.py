"""Tests for core/scintillation.py — S4 + σ_φ on synthetic signals.

Strategy: most tests construct slow-time vectors whose ground-truth
S4 / σ_φ is derivable in closed form (constant amplitude → S4=0;
amplitudes whose intensity is bipolar with equal magnitudes about a
mean gives S4 = |Δ|/mean by construction; linear phase ramp → σ_φ=0
after detrending; etc.).  Severity bins are exercised at and across
the strict-less-than boundaries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from codar_sounder.core.scintillation import (
    DEFAULT_MIN_SAMPLES,
    MAD_REJECTION_K,
    S4_EVENT_THRESHOLD,
    S4_MODERATE_MAX,
    S4_WEAK_MAX,
    SIGMA_PHI_EVENT_THRESHOLD,
    SIGMA_PHI_MODERATE_MAX,
    SIGMA_PHI_WEAK_MAX,
    ScintillationResult,
    _s4_severity,
    _sigma_phi_severity,
    compute_scintillation,
)


SRF_HZ = 1.0          # CODAR default; one slow-time sample per second.
N_SAMPLES = 60        # CODAR default CPI = 60 s × 1 Hz.

# Comfortable offset for "near-boundary" tests, well above complex64
# input precision (~1e-7) and the ~1e-6 detrend residual on large
# unwrapped phase ramps.  At 1e-4 we can construct inputs whose
# measured S4/σ_φ lands on the intended side of the boundary
# deterministically.
_BOUNDARY_TOL = 1e-4


def _complex_from(
    amplitude: np.ndarray, phase: np.ndarray,
    *, dtype=np.complex64,
) -> np.ndarray:
    """Build a complex slow-time vector from amplitude + phase arrays.

    Default ``dtype=complex64`` mirrors the production pipeline's
    range_spectrum dtype.  Tests that need higher precision (boundary
    accuracy beyond ~1e-6) can pass ``dtype=complex128``.
    """
    return (amplitude * np.exp(1j * phase)).astype(dtype)


def _phase_pattern_orthogonal_to_quadratic(
    target_std: float, n: int = N_SAMPLES,
) -> np.ndarray:
    """Phase fluctuation pattern with std == target_std, orthogonal to
    polynomials of degree ≤ 2 in t.

    v0.5.2 switched to quadratic detrend, so the test pattern must
    have Σ u_i = 0, Σ (i · u_i) = 0, AND Σ (i² · u_i) = 0 over each
    period so polyfit(deg=2) is a no-op on it.

    Solving for a period-4 pattern ``[a, b, c, d]`` with all three
    moment sums vanishing and variance = 1: ``u = (1/√5)·[-1, 3, -3, 1]``.

    Each 4-block contributes 0 to each moment, so the property
    extends to any n divisible by 4 (N_SAMPLES = 60 qualifies).
    """
    if n % 4:
        raise ValueError(f"n must be divisible by 4 for the pattern; got {n}")
    base = np.tile([-1.0, +3.0, -3.0, +1.0], n // 4) / np.sqrt(5.0)
    return target_std * base


# Backwards-compat alias for any callers that still reference the old
# name.  The new pattern is strictly stronger (orthogonal to a wider
# basis) so existing tests pass with it.
_phase_pattern_orthogonal_to_linear = _phase_pattern_orthogonal_to_quadratic


def _amplitudes_with_target_s4(
    target_s4: float, mean: float = 1.0, n: int = N_SAMPLES,
) -> np.ndarray:
    """Construct amplitudes whose intensity has S4 = target_s4 exactly.

    Two construction strategies depending on the target:

    * **target_s4 ≤ 1.0**: bipolar around the mean — intensity =
      ``mean + delta·(+1, -1, +1, -1, ...)`` with delta = target_s4·mean.
      Yields S4 = delta/mean exactly, mean(I) = mean exactly.
      Constraint: delta ≤ mean (otherwise intensity goes negative).
    * **target_s4 > 1.0** (saturated scintillation): two-level
      distribution with n_high large samples and n_low zeros.  Pick
      n_high : n_low = 1 : target_s4² so var/mean² = n_low/n_high =
      target_s4² exactly.  Slight rounding error from forcing
      n_high to be integer (negligible at n = 60).
    """
    if n % 2:
        raise ValueError("n must be even for the bipolar construction")
    if target_s4 <= 1.0:
        delta = target_s4 * mean
        signs = np.tile([+1.0, -1.0], n // 2)
        intensity = mean + delta * signs
        return np.sqrt(intensity)
    # Saturated branch.
    n_high = max(1, int(round(n / (1.0 + target_s4 ** 2))))
    intensity = np.zeros(n, dtype=np.float64)
    # mean = (n_high·a + n_low·0) / n = mean → a = mean·n/n_high.
    intensity[:n_high] = mean * n / n_high
    return np.sqrt(intensity)


class TestPureCWBaseline:
    """A pure CW signal has S4 ≈ 0 and σ_φ ≈ 0 — the canonical 'no
    scintillation' check.  Use this to validate the floor numerically."""

    def test_constant_amplitude_and_phase(self):
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_index == pytest.approx(0.0, abs=1e-10)
        assert r.sigma_phi_rad == pytest.approx(0.0, abs=1e-10)
        assert r.s4_severity == "weak"
        assert r.sigma_phi_severity == "weak"
        assert r.scintillation_event is False
        assert r.n_samples == N_SAMPLES
        assert r.confidence == pytest.approx(1.0)
        assert r.mode_doppler_hz == pytest.approx(0.0, abs=1e-10)

    def test_constant_amplitude_with_doppler_phase_ramp(self):
        """A linear phase ramp == constant Doppler shift.  σ_φ should
        be ≈ 0 after detrending; mode_doppler_hz should recover the
        imposed Doppler.

        Tolerance is set to 1e-5 rad to reflect complex64 angle
        precision (~1e-7) accumulated across an 18-rad phase ramp.
        """
        f_doppler_hz = 0.05
        t = np.arange(N_SAMPLES) / SRF_HZ
        phase = 2.0 * np.pi * f_doppler_hz * t
        z = _complex_from(np.ones_like(t), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_index == pytest.approx(0.0, abs=1e-6)
        assert r.sigma_phi_rad == pytest.approx(0.0, abs=1e-5)
        assert r.mode_doppler_hz == pytest.approx(f_doppler_hz, abs=1e-7)


class TestS4ClosedForm:
    """Construct intensity by hand so S4 equals a known target."""

    @pytest.mark.parametrize("target_s4", [0.05, 0.15, 0.25, 0.45, 0.75, 0.95])
    def test_s4_recovers_constructed_target(self, target_s4):
        amp = _amplitudes_with_target_s4(target_s4)
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_index == pytest.approx(target_s4, rel=1e-6)

    def test_s4_above_unity_is_not_clipped(self):
        """ITU-R explicitly allows S4 > 1.0 (saturated scintillation)."""
        # Intensity (mean=1, +0/+2 split) → std = 1.0 → S4 = 1.0; push
        # further with a wider split.  Mean must stay positive.
        n = N_SAMPLES
        intensity = np.where(np.arange(n) % 2 == 0, 0.05, 1.95)
        amp = np.sqrt(intensity)
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_index > 0.9


class TestSigmaPhiClosedForm:
    """Phase fluctuations with a known stdev → σ_φ recovers that stdev
    (after linear detrending removes the constant Doppler term).

    We use the period-4 ``[+1,-1,-1,+1]`` phase pattern (see
    :func:`_phase_pattern_orthogonal_to_linear`) so the detrending step
    is a true no-op by construction — the closed-form expectation
    σ_φ = target_sigma holds to complex64 input precision.
    """

    @pytest.mark.parametrize("target_sigma", [0.05, 0.15, 0.25, 0.45, 0.75])
    def test_sigma_phi_recovers_constructed_stdev(self, target_sigma):
        phase = _phase_pattern_orthogonal_to_linear(target_sigma)
        z = _complex_from(np.ones(N_SAMPLES), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.sigma_phi_rad == pytest.approx(target_sigma, abs=1e-5)

    def test_sigma_phi_doppler_trend_does_not_contaminate(self):
        """Linear phase ramp added on top of a fluctuation → ramp
        absorbed by detrending, fluctuation survives.

        Tolerance is set to 2e-2 rad to reflect complex64 precision
        accumulating over the ~38 rad total phase range (0.1 Hz × 60 s)
        — the float32 mantissa rounds large absolute phases at ~5e-6
        rad per sample, and the systematic component of that
        rounding breaks the test pattern's orthogonality slightly.
        At float64 the recovery is exact.
        """
        n = N_SAMPLES
        target_sigma = 0.3
        t = np.arange(n) / SRF_HZ
        ramp = 2.0 * np.pi * 0.1 * t       # 0.1 Hz Doppler
        fluct = _phase_pattern_orthogonal_to_quadratic(target_sigma)
        z = _complex_from(np.ones(n), ramp + fluct)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.sigma_phi_rad == pytest.approx(target_sigma, abs=2e-2)
        # Same complex64 precision limit applies to the recovered
        # Doppler slope; 1e-4 = 0.1% relative error at 0.1 Hz, well
        # below any physical Doppler resolution at SRF=1 Hz.
        assert r.mode_doppler_hz == pytest.approx(0.1, abs=1e-4)


class TestS4SeverityHelper:
    """Strict-less-than bin boundaries — tested on the pure-Python
    helper at exact float64 values.  Locks down the bin policy
    independent of the numerical signal-processing path."""

    @pytest.mark.parametrize("s4,expected", [
        (0.0,                    "weak"),
        (S4_WEAK_MAX - 1e-12,    "weak"),
        (S4_WEAK_MAX,            "moderate"),     # v0.6.3: 1.0 → moderate
        (S4_MODERATE_MAX - 1e-12, "moderate"),
        (S4_MODERATE_MAX,        "strong"),       # v0.6.3: 1.5 → strong
        (2.0,                    "strong"),       # saturated
    ])
    def test_boundary(self, s4, expected):
        assert _s4_severity(s4) == expected


class TestSigmaPhiSeverityHelper:
    @pytest.mark.parametrize("sp,expected", [
        (0.0,                            "weak"),
        (SIGMA_PHI_WEAK_MAX - 1e-12,     "weak"),
        (SIGMA_PHI_WEAK_MAX,             "moderate"),  # v0.6.2: 1.5 → moderate
        (SIGMA_PHI_MODERATE_MAX - 1e-12, "moderate"),
        (SIGMA_PHI_MODERATE_MAX,         "strong"),    # v0.6.2: 2.0 → strong
        (2.5,                            "strong"),
    ])
    def test_boundary(self, sp, expected):
        assert _sigma_phi_severity(sp) == expected


class TestS4SeverityEndToEnd:
    """Validate the severity assignment lands consistently when run
    through the full ``compute_scintillation`` path.

    Test points are kept well inside each bin (not at boundaries):
    the saturated-branch construction's integer-rounding of n_high
    introduces small S4 errors (~±0.03) that make boundary-adjacent
    tests fragile.  Helper tests above cover boundary assignment at
    exact float64 values where construction error is irrelevant.
    """

    @pytest.mark.parametrize("target_s4,expected", [
        (0.05, "weak"),
        (0.50, "weak"),
        (0.95, "weak"),       # well within v0.6.3 weak bin (< 1.0)
        (1.20, "moderate"),
        (1.40, "moderate"),   # well within moderate bin (1.0–1.5)
        (1.80, "strong"),     # max saturated target the v0.5.1 MAD
                              # rejection tolerates — at target ≥ ~2.0
                              # the sparse construction has so many
                              # zeros that the per-peak MAD rejects
                              # the high-amplitude samples as outliers
                              # and returns "unknown".  Boundary
                              # assignment at exact float64 values is
                              # covered by TestS4SeverityHelper above.
    ])
    def test_e2e(self, target_s4, expected):
        amp = _amplitudes_with_target_s4(target_s4)
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_severity == expected, (
            f"target_s4={target_s4} measured S4={r.s4_index:.6f} "
            f"got severity={r.s4_severity}, expected {expected}"
        )


class TestSigmaPhiSeverityEndToEnd:
    """End-to-end measurement plumbing — verifies a constructed σ_φ
    target round-trips through ``compute_scintillation`` to the
    correct severity bin.  Restricted to the *unwrap-safe* regime
    (target σ ≲ 1.0): the orthogonal-to-quadratic phase pattern's
    per-sample step grows with target σ, and exceeds π/sample
    around target σ ≈ 1.05 — at which point ``np.unwrap`` interprets
    the swing as a 2π wrap, corrupting the test signal.

    Boundary assignment at the new v0.6.2 thresholds (1.5 / 2.0) is
    covered by ``TestSigmaPhiSeverityHelper`` at exact float64 values
    where unwrap is irrelevant.  Real production σ_φ ≈ 1.3 rad on
    quiet days is measured from naturally-distributed phase that
    doesn't violate unwrap per-sample even though its std is large.
    """

    @pytest.mark.parametrize("target_sp,expected", [
        (0.05, "weak"),
        (0.3,  "weak"),
        (0.7,  "weak"),     # well within the new weak bin (< 1.5)
        (1.0,  "weak"),     # still weak under v0.6.2 thresholds
    ])
    def test_e2e_unwrap_safe(self, target_sp, expected):
        phase = _phase_pattern_orthogonal_to_linear(target_sp)
        z = _complex_from(np.ones(N_SAMPLES), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.sigma_phi_severity == expected, (
            f"target_sp={target_sp} measured σ_φ={r.sigma_phi_rad:.6f} "
            f"got severity={r.sigma_phi_severity}, expected {expected}"
        )


class TestScintillationEvent:
    """Event gate fires iff S4 ≥ 0.3 or σ_φ ≥ 0.2 (matches the
    lower bound of the moderate bin in either index)."""

    def test_no_event_when_both_weak(self):
        amp = _amplitudes_with_target_s4(0.1)
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.scintillation_event is False

    def test_event_when_s4_above_threshold(self):
        # Target safely above the event threshold (v0.6.3: S4 ≥ 1.0).
        # Saturated-branch construction has small integer-rounding
        # error (~±0.03), so a comfortable margin avoids fragility.
        amp = _amplitudes_with_target_s4(1.2)
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_index >= S4_EVENT_THRESHOLD
        assert r.scintillation_event is True

    def test_event_when_sigma_phi_at_threshold(self):
        # The σ_φ event threshold (v0.6.2: 1.5 rad) sits above the
        # unwrap-safe regime of the test pattern (target σ ≲ 1.0).
        # Construct phases directly and bypass the orthogonal-quad
        # pattern + complex round-trip — the event gate is a simple
        # `>=` check on the computed σ_φ, fully covered by the helper
        # tests above.  Here we verify the gate fires for a manually
        # constructed real-but-noisy slow_time vector whose σ_φ
        # exceeds threshold.
        rng = np.random.default_rng(seed=2026)
        n = N_SAMPLES
        # Random phases drawn uniformly in (-π, π); std ≈ π/√3 ≈ 1.81
        # > 1.5 event threshold.
        phase = rng.uniform(-np.pi, np.pi, size=n)
        z = _complex_from(np.ones(n), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.sigma_phi_rad >= SIGMA_PHI_EVENT_THRESHOLD
        assert r.scintillation_event is True


class TestQualityGating:

    def test_below_min_samples_returns_unknown(self):
        z = np.ones(DEFAULT_MIN_SAMPLES - 1, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_severity == "unknown"
        assert r.sigma_phi_severity == "unknown"
        assert r.confidence == 0.0
        assert r.scintillation_event is False
        assert r.n_samples == DEFAULT_MIN_SAMPLES - 1

    def test_at_min_samples_classifies(self):
        z = np.ones(DEFAULT_MIN_SAMPLES, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_severity == "weak"
        assert r.confidence > 0.0

    def test_zero_signal_returns_unknown(self):
        """Range bin sitting in a clutter-mask null → all-zero
        complex vector → refuse to classify."""
        z = np.zeros(N_SAMPLES, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_severity == "unknown"
        assert r.sigma_phi_severity == "unknown"
        assert r.confidence == 0.0
        assert r.scintillation_event is False

    def test_confidence_saturates_at_30_samples(self):
        z = np.ones(30, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.confidence == pytest.approx(1.0)

    def test_confidence_scales_below_saturation(self):
        # 15 samples → 0.5
        z = np.ones(15, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.confidence == pytest.approx(0.5, abs=1e-9)

    def test_invalid_sample_rate_raises(self):
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        with pytest.raises(ValueError):
            compute_scintillation(z, sample_rate_hz=0.0)
        with pytest.raises(ValueError):
            compute_scintillation(z, sample_rate_hz=-1.0)

    def test_nan_input_yields_unknown(self):
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        z[5] = complex(np.nan, np.nan)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        # nan propagates through np.var / np.std → s4_index NaN →
        # unknown-result fallback.
        assert r.s4_severity == "unknown"
        assert r.confidence == 0.0

    def test_inf_input_yields_unknown(self):
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        z[5] = complex(np.inf, 0.0)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_severity == "unknown"
        assert r.confidence == 0.0


class TestSampleRateHandling:
    """mode_doppler_hz must scale with sample_rate_hz, but S4/σ_φ are
    sample-rate-independent (they're per-sample statistics)."""

    def test_mode_doppler_uses_sample_rate(self):
        # Same phase ramp across N samples at two different cadences
        # → recovered Doppler differs by the rate ratio.
        n = N_SAMPLES
        # 2π·n_samples = 1 cycle/(n_samples/sample_rate_hz) Hz.
        phase = 2.0 * np.pi * np.arange(n) / n
        z = _complex_from(np.ones(n), phase)

        r1 = compute_scintillation(z, sample_rate_hz=1.0)
        r2 = compute_scintillation(z, sample_rate_hz=10.0)
        assert r2.mode_doppler_hz == pytest.approx(
            10.0 * r1.mode_doppler_hz, rel=1e-6
        )

    def test_s4_independent_of_sample_rate(self):
        amp = _amplitudes_with_target_s4(0.4)
        z = _complex_from(amp, np.zeros_like(amp))
        r1 = compute_scintillation(z, sample_rate_hz=1.0)
        r2 = compute_scintillation(z, sample_rate_hz=10.0)
        assert r1.s4_index == pytest.approx(r2.s4_index, rel=1e-9)


class TestMADOutlierRejection:
    """v0.5.1: one bad-sweep contamination must not corrupt S4/σ_φ on
    the remaining clean samples."""

    def test_single_outlier_rejected(self):
        """Inject one 5×-magnitude outlier into an otherwise-clean
        slow-time vector; the rejected sample doesn't get to inflate
        S4."""
        # Clean baseline: constant amplitude, zero phase → S4 = 0.
        n = N_SAMPLES
        amp = np.ones(n, dtype=np.float64)
        # Insert a single big spike.
        amp[7] = 5.0
        z = _complex_from(amp, np.zeros(n))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.n_outliers_rejected == 1
        # On the cleaned 59 samples (M=60 default), all intensity = 1
        # → S4 = 0.
        assert r.s4_index == pytest.approx(0.0, abs=1e-10)
        assert r.s4_severity == "weak"
        assert r.n_samples == n - 1

    def test_multiple_outliers_rejected(self):
        n = N_SAMPLES
        amp = np.ones(n, dtype=np.float64)
        # 4 spikes scattered across the vector.
        for idx in (3, 11, 27, 44):
            amp[idx] = 6.0
        z = _complex_from(amp, np.zeros(n))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.n_outliers_rejected == 4
        assert r.s4_index == pytest.approx(0.0, abs=1e-10)
        assert r.n_samples == n - 4

    def test_no_outliers_on_clean_signal(self):
        """A signal with intensity uniformly distributed within a
        narrow band — no MAD-defined outliers."""
        n = N_SAMPLES
        amp = _amplitudes_with_target_s4(0.2)
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        # Construction is symmetric ±0.2 about mean = 1, so all samples
        # land within K · MAD of median.
        assert r.n_outliers_rejected == 0
        assert r.n_samples == n
        assert r.s4_index == pytest.approx(0.2, abs=1e-4)

    def test_zero_mad_skips_rejection(self):
        """Perfectly uniform input → MAD = 0 → filter must be a
        no-op (don't divide by zero or reject everything)."""
        n = N_SAMPLES
        z = np.ones(n, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.n_outliers_rejected == 0
        assert r.n_samples == n

    def test_rejection_drops_below_min_samples_returns_unknown(self):
        """If outlier rejection leaves fewer than ``min_samples``
        clean samples, the classifier should refuse to classify.

        Use 11 clean + 1 outlier with min_samples=12 so any rejection
        lands us below the floor.  Avoids the 33%-contamination
        regime where the MeanAD-fallback threshold opens too wide to
        reject (a real limitation of robust statistics — breakdown
        point ≈ 50% for pure MAD, lower with MeanAD fallback).
        """
        amp = np.ones(12, dtype=np.float64)
        amp[6] = 100.0     # one big outlier among 11 clean samples
        z = _complex_from(amp, np.zeros_like(amp))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ, min_samples=12)
        assert r.s4_severity == "unknown"
        assert r.sigma_phi_severity == "unknown"
        assert r.n_outliers_rejected == 1
        assert r.n_samples == 11
        # 11 retained < min_samples=12 → unknown.

    def test_pre_rejected_mask_excludes_upstream_zeros(self):
        """v0.6.1: when dechirp's per-sweep MAD pre-filter has already
        zeroed some sweeps, scintillation should treat those positions
        as already-rejected rather than re-running MAD on them
        (zeros pollute the MAD scale and obscure the zeros' own
        outlier status)."""
        n = 15
        # Clean baseline with natural variation; zero out 2 positions
        # as if dechirp had pre-filtered them.
        rng = np.random.default_rng(seed=11)
        amp = 1.0 + 0.1 * rng.standard_normal(n)
        amp[2] = 0.0
        amp[9] = 0.0
        pre_mask = np.zeros(n, dtype=bool)
        pre_mask[2] = True
        pre_mask[9] = True

        z = _complex_from(amp, np.zeros(n))
        r = compute_scintillation(
            z, sample_rate_hz=SRF_HZ, pre_rejected_mask=pre_mask,
        )
        # n_samples = upstream-kept (13) minus any per-peak MAD reject.
        # The 13 clean samples have small variance, so per-peak MAD
        # should find no additional outliers → n_samples = 13.
        assert r.n_samples == 13
        # n_outliers_rejected counts only THIS function's work,
        # i.e. *additional* rejections beyond the upstream mask.
        assert r.n_outliers_rejected == 0
        # The result is computed on the 13 clean samples only — S4
        # should be tiny (no contamination).
        assert r.s4_index < 0.2
        assert r.s4_severity == "weak"

    def test_pre_rejected_mask_plus_additional_outlier(self):
        """Upstream mask + a per-peak outlier that survived → both
        rejection mechanisms fire; n_outliers_rejected counts ONLY
        the extra per-peak rejection."""
        n = 15
        amp = np.ones(n, dtype=np.float64)
        amp[3] = 0.0      # upstream-zeroed
        amp[7] = 5.0      # per-peak outlier (a survived intensity spike)
        pre_mask = np.zeros(n, dtype=bool)
        pre_mask[3] = True
        z = _complex_from(amp, np.zeros(n))
        r = compute_scintillation(
            z, sample_rate_hz=SRF_HZ, pre_rejected_mask=pre_mask,
        )
        # Per-peak MAD on the 14 upstream-kept samples should catch
        # the spike at index 7.
        assert r.n_outliers_rejected == 1
        assert r.n_samples == 13     # 15 - 1 (upstream) - 1 (per-peak) = 13

    def test_pre_rejected_mask_dimension_mismatch_raises(self):
        z = np.ones(15, dtype=np.complex64)
        bad_mask = np.zeros(14, dtype=bool)
        with pytest.raises(ValueError, match="must match"):
            compute_scintillation(
                z, sample_rate_hz=SRF_HZ, pre_rejected_mask=bad_mask,
            )

    def test_field_simulation_one_bad_sweep_per_15(self):
        """Reproduce the 2026-05-21 live-verification finding:
        14 clean coherent sweeps + 1 spike at the same index in every
        peak's slow-time vector.  Without MAD rejection this would
        give S4 ≈ 3 (false strong event).  With it, S4 should be
        small."""
        n = 15
        amp = np.full(n, 0.035, dtype=np.float64)   # clean baseline
        amp[6] = 0.232                              # one bad sweep (5×)
        z = _complex_from(amp, np.zeros(n))
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.n_outliers_rejected == 1
        assert r.s4_severity == "weak"
        assert r.s4_index < 0.05
        assert r.scintillation_event is False


class TestResultDataclass:
    """Lock the field set — schema-drift catch.  If you add or rename a
    field, this test should be updated deliberately."""

    def test_field_set(self):
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert isinstance(r, ScintillationResult)
        # Fields in dataclass order.
        expected = {
            "s4_index", "s4_severity",
            "sigma_phi_rad", "sigma_phi_severity",
            "scintillation_event", "confidence",
            "n_samples", "n_outliers_rejected",
            "mode_doppler_hz",
            # v0.6 diagnostics:
            "sigma_phi_linear_rad", "sigma_phi_quadratic_rad",
            "sigma_phi_underfit_ratio",
        }
        assert set(r.__dataclass_fields__.keys()) == expected


class TestUnderfitRatio:
    """v0.6: sigma_phi_underfit_ratio = sigma_phi_linear / sigma_phi_quad.
    Equals 1 for clean signals with at most constant Doppler;
    grows above 1 when phase has curvature the linear detrend
    can't capture."""

    def test_constant_amplitude_zero_phase_ratio_is_unity(self):
        """A pure CW signal has zero σ_φ under both estimators →
        ratio defaults to 1.0 by convention (avoid div-by-zero)."""
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.sigma_phi_underfit_ratio == pytest.approx(1.0)

    def test_constant_doppler_ratio_is_unity(self):
        """Linear phase ramp (constant Doppler, no curvature) →
        linear detrend is perfect → linear residual == quadratic
        residual → ratio == 1 (within complex64 precision)."""
        n = N_SAMPLES
        f_doppler_hz = 0.05
        t = np.arange(n) / SRF_HZ
        phase = 2.0 * np.pi * f_doppler_hz * t
        z = _complex_from(np.ones_like(t), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        # Both σ_φ values are near-zero floor; the ratio of two near-
        # zero numbers is noisy, so the test asserts both are tiny
        # rather than checking the ratio directly.
        assert r.sigma_phi_linear_rad < 1e-4
        assert r.sigma_phi_quadratic_rad < 1e-4

    def test_quadratic_phase_inflates_linear_residual(self):
        """A purely quadratic phase term should give a *huge* linear-
        detrend residual (linear can't remove curvature) and a
        near-zero quadratic-detrend residual → underfit ratio >> 1."""
        n = N_SAMPLES
        t = np.arange(n, dtype=np.float64)
        # Pure t² phase: large coefficient so the curvature stands out
        # above complex64 noise.
        alpha = 0.005       # rad/sample²
        # Center around mid-CPI so max phase stays modest (avoid
        # complex64 precision degradation at large absolute phases).
        t_centered = t - t.mean()
        phase = alpha * t_centered ** 2
        z = _complex_from(np.ones(n), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        # Quadratic should absorb it cleanly; linear leaves big
        # residual.
        assert r.sigma_phi_quadratic_rad < 0.05
        assert r.sigma_phi_linear_rad > 0.5
        assert r.sigma_phi_underfit_ratio > 5.0

    def test_canonical_field_equals_quadratic(self):
        """sigma_phi_rad == sigma_phi_quadratic_rad — the canonical
        field used for severity classification is by construction the
        quadratic-detrend value."""
        n = N_SAMPLES
        phase = _phase_pattern_orthogonal_to_quadratic(0.3)
        z = _complex_from(np.ones(n), phase)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.sigma_phi_rad == r.sigma_phi_quadratic_rad

    def test_ratio_at_least_one(self):
        """By construction the quadratic basis ⊇ linear basis so
        quadratic residual ≤ linear residual → ratio ≥ 1 for any
        non-degenerate input."""
        # A few different inputs.
        rng = np.random.default_rng(seed=7)
        for _ in range(10):
            phase = rng.standard_normal(N_SAMPLES) * 0.5
            z = _complex_from(np.ones(N_SAMPLES), phase)
            r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
            # Allow tiny float epsilon under 1.0 (≤ 1e-9 numerical
            # noise) but reject anything genuinely below 1.
            assert r.sigma_phi_underfit_ratio >= 1.0 - 1e-9, (
                f"ratio={r.sigma_phi_underfit_ratio} "
                f"lin={r.sigma_phi_linear_rad} "
                f"quad={r.sigma_phi_quadratic_rad}"
            )

    def test_unknown_result_ratio_is_unity(self):
        """The unknown-result fallback returns ratio=1.0."""
        z = np.ones(DEFAULT_MIN_SAMPLES - 1, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        assert r.s4_severity == "unknown"
        assert r.sigma_phi_underfit_ratio == 1.0

    def test_frozen(self):
        z = np.ones(N_SAMPLES, dtype=np.complex64)
        r = compute_scintillation(z, sample_rate_hz=SRF_HZ)
        with pytest.raises(Exception):  # FrozenInstanceError (or dataclasses-specific)
            r.s4_index = 0.5            # type: ignore[misc]

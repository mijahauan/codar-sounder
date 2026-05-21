"""Ionospheric scintillation indices — S4 (amplitude) and σ_φ (phase).

Per ITU-R Recommendation P.531 (Ionospheric propagation data and
prediction methods required for the design of satellite services and
systems).  The same conventions are used by GNSS ionospheric monitoring
networks and by hf-timestd's WWV-tone scintillation path
(``hf_timestd/core/advanced_signal_analysis.py`` lines 880-1000) — this
module is the codar-sounder analogue for *oblique* propagation, applied
per propagation mode (1F2 high-ray, 1F2 low-ray, E, Es).

Math
----
Given a complex baseband time series ``z(t) = A(t)·exp(jφ(t))`` at a
fixed propagation mode (e.g. one range bin's slow-time vector across a
single CPI):

    Intensity         I(t) = |z(t)|² = A(t)²
    S4 (amplitude)    S4   = √( Var(I) / ⟨I⟩² )
    σ_φ (phase)       σ_φ  = std( φ_detrended )

where ``φ_detrended`` is the unwrapped phase with a linear-in-time
trend removed (the trend = ionospheric Doppler shift of the mode; what
remains is the scintillation fluctuation).

Severity bins (strict-less-than):

    S4   < 1.0 → weak     | σ_φ < 1.5 → weak    (v0.6.3: both HF-cal)
    S4   < 1.5 → moderate | σ_φ < 2.0 → moderate
    S4   ≥ 1.5 → strong   | σ_φ ≥ 2.0 → strong

A "scintillation event" is declared when ``S4 ≥ 1.0 or σ_φ ≥ 1.5``.

Both S4 and σ_φ thresholds depart from ITU-R P.531 canonical
(S4: 0.3 / 0.6; σ_φ: 0.2 / 0.5) because those values were
calibrated for narrowband single-mode signals (GNSS, SHF
satellite) where the *intrinsic* multipath floor is much lower
than HF oblique.  Cross-comparisons to GNSS scintillation
literature should treat the codar-sounder values as
HF-recalibrated; absolute numbers (the float ``s4_index`` and
``sigma_phi_rad`` fields) remain comparable to other HF
multipath sounders but NOT to GNSS at face value.

Calibration history
-------------------
The thresholds were tightened during 2026-05-21 live verification on
bee1-rx888 SEAB (13.45 MHz, 1416 km):

  v0.5.0 (initial)        ITU-R: S4 0.3/0.6  σ_φ 0.2/0.5
  v0.5.2 (HF empirical)   HF:    S4 0.3/0.6  σ_φ 0.5/1.0
  v0.6.2 (Kp-validated)   HF:    S4 0.3/0.6  σ_φ 1.5/2.0
  v0.6.3 (both Kp-cal)    HF:    S4 1.0/1.5  σ_φ 1.5/2.0

v0.6.2 fixed σ_φ but left S4 ITU-R-canonical — live data
immediately showed scintillation_event rate still 94% because S4
was now alone driving the events.  Quiet-day S4 distribution on
2026-05-21 (Kp 1.0-3.0; 11,577 records):

  p10  = 0.56     # HF Rayleigh-fading baseline
  p50  = 0.78     # median peak on a quiet day
  p90  = 1.05
  p95  = 1.30

The median peak's S4 = 0.78 sits well above ITU-R's "strong"
threshold (0.6) on a quiet day — same story as σ_φ.  At HF oblique
with multipath, the signal Rayleigh-fades and produces S4 ≈ 0.7-1.0
by construction with no real scintillation.  v0.6.3 thresholds
treat the p90 of quiet-day data (~1.0) as the weak/moderate
boundary, and reserve "strong" for ≥ 1.5 — well above any quiet-
day observation.

Final calibration awaits a Kp ≥ 5 storm with v0.5+ logging.
Expect another nudge if storm-day statistics suggest further
adjustment.

Cadence caveat
--------------
At codar-sounder's default CPI = 60 s with sweep-repetition rate 1 Hz,
``compute_scintillation`` operates on M=60 slow-time samples.  The
quadratic detrend (v0.5.2 — was linear in v0.5/0.5.1) acts as an
effective high-pass at 1/CPI ≈ 0.017 Hz — *not* ITU-R's canonical
0.1 Hz (which would require a higher cadence than 1 Hz to be useful
in this band).  Downstream consumers cross-comparing to GNSS σ_φ
should be aware.

The detrending degree changed from 1 → 2 in v0.5.2 after live data
showed that real F-region oblique paths produce *curved* slow-time
phase trajectories that a linear fit cannot track.  Concretely, on
SEAB peaks at M=15:

   peak  σ_φ_linear   σ_φ_quadratic   reduction
    0    0.485        0.454            6%
    1    0.895        0.856            4%
    2    1.548        0.622           60%   ← linear was underfitting
    3    1.386        0.954           31%

Quadratic captures TID-scale and beat-multipath curvature without
over-fitting genuine 5-30 s scintillation (which has higher
frequency content than a quadratic can absorb).

Quality gating
--------------
- ``n_samples < min_samples`` → severities = ``"unknown"``,
  ``confidence = 0``.  The default ``min_samples = 10`` is a hard
  statistical floor (CV of variance estimate ~ 30%).
- ``mean_intensity < 1e-30`` (range bin sits in a clutter-mask null
  or has zero signal) → same unknown result, regardless of sample
  count.
- Any non-finite output (NaN/Inf) → confidence forced to 0.

Outlier rejection (v0.5.1)
--------------------------
Live verification on bee1-rx888 (2026-05-21) revealed that a single
contaminated sweep per CPI — broadband spectral leakage from one
unusable matched-filter output, typically with its FFT peak in the
negative-range half — was placing one anomalously-large intensity
sample into every range bin's M-vector.  At M=15 (the production
CPI=15s × SRF=1Hz) one outlier produces S4 ≈ √(M-1) ≈ 3.7 and
σ_φ ≈ π/√3, falsely flagging every peak as a strong scintillation
event.

Robust intensity filtering with the median absolute deviation
removes the contamination: drop samples whose
``|I_k - median(I)| > MAD_REJECTION_K · MAD(I)`` before computing
S4 and σ_φ.  ``MAD_REJECTION_K = 4.0`` is comfortably above the
Gaussian-equivalent 3σ (real Rayleigh-distributed scintillation
produces intensity tails out to ~3·median; bad sweeps produce
~10×).  The filter is a no-op when MAD = 0 (uniform input).
``n_outliers_rejected`` is exposed on the result so the rejection
rate can be tracked at the sink.

Confidence model is intentionally lean: ``confidence =
min(1, n_samples/30)`` clipped to [0, 1].  hf-timestd's
``outlier_factor`` (penalising high coefficient-of-variation) was
omitted here because it is inversely correlated with S4 itself — it
would suppress the confidence of *strong real* scintillation events,
which is the opposite of what scintillation monitoring wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# S4 severity-bin boundaries — HF-recalibrated (v0.6.3; see module
# docstring).  ITU-R single-mode thresholds (0.3 / 0.6) misclassify
# the HF Rayleigh-fading baseline as "strong" by construction —
# median quiet-day S4 at SEAB / 13.45 MHz / 1416 km is 0.78.
S4_WEAK_MAX = 1.0
S4_MODERATE_MAX = 1.5

# σ_φ severity-bin boundaries — Kp-validated (v0.6.2; see module
# docstring).  Calibrated from the 2026-05-21 Kp-correlation analysis
# which showed σ_φ ≈ 1.27 rad at Kp=1.00 (very quiet) — the HF
# intrinsic floor at SEAB / 13.45 MHz / 1416 km sits at ~1.2-1.5 rad,
# well above v0.5.2's 0.5/1.0 thresholds.
SIGMA_PHI_WEAK_MAX = 1.5
SIGMA_PHI_MODERATE_MAX = 2.0

# "Event" gate: matches the lower bound of the moderate bin in
# either index — a clean, monitoring-actionable threshold.  Both
# thresholds are now HF-calibrated (S4 ≥ 1.0 OR σ_φ ≥ 1.5).
S4_EVENT_THRESHOLD = 1.0
SIGMA_PHI_EVENT_THRESHOLD = 1.5

# Default hard floor on slow-time samples.  60 samples (the codar-sounder
# default at CPI=60s, SRF=1Hz) sits well above this; the floor protects
# downscaled CPIs from producing meaningless statistics.
DEFAULT_MIN_SAMPLES = 10

# Sample-count at which confidence saturates to 1.0.  30 samples gives
# variance-estimate CV ≈ 18%, comparable to ITU-R's per-minute S4 spec.
CONFIDENCE_SAMPLE_SATURATION = 30.0

# Below this intensity we treat the range bin as "no signal" and refuse
# to classify.  1e-30 sits well below any realistic numerical noise floor
# (complex64 dynamic range is ~1e-38 .. 1e+38) — it only fires when the
# slow-time vector is identically zero or near-zero.
INTENSITY_NULL_THRESHOLD = 1e-30

# MAD-based outlier rejection threshold (v0.5.1).  Samples whose
# |I_k - median(I)| > MAD_REJECTION_K · MAD(I) are dropped before the
# variance/std calculation.  4 is comfortably above Gaussian-equivalent
# 3σ; Rayleigh-distributed real scintillation produces intensity tails
# to ~3·median while bad-sweep leakage produces ~10×.
MAD_REJECTION_K = 4.0


@dataclass(frozen=True)
class ScintillationResult:
    """Per-CPI per-mode scintillation indices.

    Fields:
        s4_index: amplitude scintillation index, dimensionless ≥ 0.
            Values > 1.0 are valid (saturated scintillation) — do not
            clip.
        s4_severity: ``"weak"`` / ``"moderate"`` / ``"strong"`` /
            ``"unknown"`` per ITU-R P.531 bins.
        sigma_phi_rad: phase scintillation index, radians ≥ 0.
        sigma_phi_severity: same bins as ``s4_severity``.
        scintillation_event: ``True`` when S4 ≥ 0.3 or σ_φ ≥ 0.2.
        confidence: 0..1; saturates at 30 finite samples after outlier
            rejection.  0 implies ``"unknown"`` severities.
        n_samples: number of slow-time samples *retained* (after MAD
            outlier rejection).
        n_outliers_rejected: count of samples dropped by the MAD filter
            (v0.5.1).  0 means the slow-time vector was clean; high
            values indicate bad-sweep contamination.
        mode_doppler_hz: linear-in-time phase slope removed during
            detrending, expressed as Hz.  This is the propagation mode's
            ionospheric Doppler shift; useful as a stand-alone diagnostic
            (a TID's signature is a slow drift in this number).
        sigma_phi_linear_rad: σ_φ computed from a *linear* polyfit
            detrend (v0.6 diagnostic).  Always ≥ ``sigma_phi_rad``
            (quadratic) because the linear basis is a subspace of the
            quadratic basis.
        sigma_phi_quadratic_rad: σ_φ computed from a *quadratic*
            polyfit detrend.  Numerically identical to
            ``sigma_phi_rad`` — kept as a named field for symmetry
            with ``sigma_phi_linear_rad`` so downstream readers don't
            have to know that the canonical bin uses the quadratic
            value.
        sigma_phi_underfit_ratio: ``sigma_phi_linear_rad /
            sigma_phi_quadratic_rad`` (or 1.0 if quadratic is zero).
            Equals 1.0 when the slow-time phase has no curvature
            beyond constant Doppler (clean single-mode propagation);
            >> 1 when residual phase curvature exists — TIDs,
            multipath beating, accelerating ionospheric Doppler.
            Useful as a TID detector independent of σ_φ severity.
    """
    s4_index: float
    s4_severity: str
    sigma_phi_rad: float
    sigma_phi_severity: str
    scintillation_event: bool
    confidence: float
    n_samples: int
    n_outliers_rejected: int
    mode_doppler_hz: float
    sigma_phi_linear_rad: float
    sigma_phi_quadratic_rad: float
    sigma_phi_underfit_ratio: float


def _s4_severity(s4_index: float) -> str:
    if s4_index < S4_WEAK_MAX:
        return "weak"
    if s4_index < S4_MODERATE_MAX:
        return "moderate"
    return "strong"


def _sigma_phi_severity(sigma_phi_rad: float) -> str:
    if sigma_phi_rad < SIGMA_PHI_WEAK_MAX:
        return "weak"
    if sigma_phi_rad < SIGMA_PHI_MODERATE_MAX:
        return "moderate"
    return "strong"


def _unknown_result(
    n_samples: int, n_outliers_rejected: int = 0,
) -> ScintillationResult:
    return ScintillationResult(
        s4_index=0.0,
        s4_severity="unknown",
        sigma_phi_rad=0.0,
        sigma_phi_severity="unknown",
        scintillation_event=False,
        confidence=0.0,
        n_samples=n_samples,
        n_outliers_rejected=n_outliers_rejected,
        mode_doppler_hz=0.0,
        sigma_phi_linear_rad=0.0,
        sigma_phi_quadratic_rad=0.0,
        sigma_phi_underfit_ratio=1.0,
    )


def compute_scintillation(
    slow_time: np.ndarray,
    *,
    sample_rate_hz: float,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    pre_rejected_mask: Optional[np.ndarray] = None,
) -> ScintillationResult:
    """Compute S4 + σ_φ from one propagation mode's complex slow-time vector.

    Args:
        slow_time: 1-D complex array — the per-sweep matched-filter
            output at a fixed range bin (i.e. one column of the
            pre-Doppler-FFT M×N range-spectrum tensor).  At codar's
            default CPI=60s, SRF=1Hz this is 60 samples long.
        sample_rate_hz: the slow-time cadence (= sweep_repetition_hz
            for a per-CPI extraction).  Used to convert the detrended
            linear phase slope into ``mode_doppler_hz``.
        min_samples: refuse to classify below this many samples.
        pre_rejected_mask: optional bool array, same length as
            ``slow_time``.  ``True`` at sample positions already
            rejected upstream (e.g., zeroed by ``dechirp``'s per-sweep
            MAD pre-filter).  Excluded from the per-peak MAD test so
            their zeros don't pollute the MAD scale.
            ``n_outliers_rejected`` counts only the additional
            rejections this function performed, not the pre-rejected
            count.

    Returns:
        :class:`ScintillationResult`.  Never raises — pathological
        inputs (empty array, all-zero signal, NaN/Inf) yield an
        ``"unknown"`` result with ``confidence = 0``.
    """
    slow_time = np.asarray(slow_time)
    n_input = int(slow_time.size)

    if n_input < min_samples:
        return _unknown_result(n_input)
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}")

    amplitudes = np.abs(slow_time)
    intensity = amplitudes.astype(np.float64) ** 2

    # ─── Upstream rejection mask (v0.6.1) ────────────────────────────
    # Samples already zeroed by dechirp's per-sweep MAD pre-filter are
    # excluded from the per-peak MAD calculation here.  Otherwise the
    # embedded zeros would pollute the MAD scale (raising the
    # threshold so MAD-on-intensity may not catch them itself), and
    # the same sample could be implicitly double-counted between the
    # two stages.
    if pre_rejected_mask is not None:
        pre_rejected_mask = np.asarray(pre_rejected_mask, dtype=bool)
        if pre_rejected_mask.shape != (n_input,):
            raise ValueError(
                f"pre_rejected_mask shape {pre_rejected_mask.shape} "
                f"must match slow_time length {n_input}"
            )
        upstream_keep = ~pre_rejected_mask
    else:
        upstream_keep = np.ones(n_input, dtype=bool)

    # ─── MAD-based outlier rejection (v0.5.1) ────────────────────────
    # A single contaminated sweep (broadband spectral leakage from an
    # unusable matched-filter row) puts one anomalously-large
    # intensity sample into every range bin's slow-time vector.  Drop
    # samples whose intensity is more than ``MAD_REJECTION_K`` MADs
    # from the median.
    #
    # MAD=0 degeneracy: when the baseline is exact-uniform (rare in
    # real data — thermal noise ensures variation — but common in
    # synthetic tests), MAD collapses to zero and the simple filter
    # would reject nothing.  Fall back to scaled MeanAD (Iglewicz &
    # Hoaglin 1993): for Gaussian-distributed data, 1.2533·MeanAD ≈
    # MAD asymptotically, so the threshold is consistent across
    # both branches.  Both-zero (perfectly uniform input) means
    # there's nothing to reject — keep everything.
    #
    # Compute MAD statistics on the upstream-kept samples only so
    # upstream zeros don't drag the median or inflate MAD.
    intensity_for_mad = intensity[upstream_keep]
    if intensity_for_mad.size > 0:
        deviations_kept = np.abs(
            intensity_for_mad - float(np.median(intensity_for_mad))
        )
        mad_intensity = float(np.median(deviations_kept))
        if mad_intensity > 0.0:
            scale = mad_intensity
        else:
            scale = 1.2533 * float(np.mean(deviations_kept))
    else:
        scale = 0.0
    # Apply the threshold to all input positions, but only positions
    # that are upstream-kept can possibly be retained.  This
    # automatically forces zeros (upstream-rejected) out of the keep
    # set.
    if scale > 0.0:
        deviations_full = np.abs(
            intensity - float(np.median(intensity_for_mad))
        )
        keep = upstream_keep & (deviations_full <= MAD_REJECTION_K * scale)
    else:
        keep = upstream_keep.copy()
    # n_outliers_rejected counts only the *additional* rejection this
    # function performed beyond the upstream mask, so the two stages'
    # rejection counts don't double-add.
    n_outliers_rejected = int(int(upstream_keep.sum()) - int(keep.sum()))

    # Re-check the floor against the *retained* sample count — if
    # outlier rejection pushes us below min_samples, we don't have
    # enough clean data to classify.
    n_samples = int(keep.sum())
    if n_samples < min_samples:
        # Report retained count, not the input count: ``n_samples`` is
        # documented as the post-rejection retained count.
        return _unknown_result(n_samples, n_outliers_rejected)

    slow_kept = slow_time[keep]
    intensity_kept = intensity[keep]

    mean_intensity = float(np.mean(intensity_kept))
    if mean_intensity < INTENSITY_NULL_THRESHOLD:
        # Range bin has effectively no signal — refuse to classify.
        return _unknown_result(n_input, n_outliers_rejected)

    # ─── S4 ──────────────────────────────────────────────────────────
    # Population variance (ddof=0) matches the ITU-R conventional
    # definition; the difference vs. sample variance is negligible at
    # n ≥ 10 but the convention matters for reproducibility.
    intensity_variance = float(np.var(intensity_kept, ddof=0))
    s4_index = float(np.sqrt(intensity_variance) / mean_intensity)

    # ─── σ_φ ─────────────────────────────────────────────────────────
    # Unwrap from the complex signal directly (don't ask the caller to
    # pre-unwrap; np.unwrap is idempotent so this is safe even if they
    # did).  Unwrap on the *full* slow_time first so the wrap
    # bookkeeping spans the input cleanly, then index by the keep
    # mask.  Times must use the original sample indices so the
    # polyfit-derived doppler still corresponds to physical Hz at
    # ``sample_rate_hz``.
    phases_full = np.unwrap(np.angle(slow_time)).astype(np.float64)
    phases = phases_full[keep]
    sample_indices = np.arange(n_input, dtype=np.float64)[keep]
    times = sample_indices / sample_rate_hz

    # Detrend (v0.5.2): quadratic is canonical for severity bins —
    # captures TID-scale and beat-multipath phase curvature that the
    # live bee1-rx888 data showed linear could not track.  Center the
    # time axis before fitting so the *linear* coefficient is the
    # average-Doppler slope at the CPI centroid.
    #
    # v0.6 adds the linear-detrend σ_φ as a diagnostic field so a
    # downstream consumer can compute the *underfit ratio* (linear /
    # quadratic) as a TID/multipath-beating signature, independent of
    # the σ_φ severity classification.
    try:
        times_centered = times - float(np.mean(times))
        # Quadratic — canonical.  polyfit returns coefficients highest-
        # degree first: [a (rad/s²), b (rad/s), c (rad)] for
        # phase = a·τ² + b·τ + c with τ = times_centered.
        coeffs_quad = np.polyfit(times_centered, phases, deg=2)
        mode_doppler_hz = float(coeffs_quad[1] / (2.0 * np.pi))
        phase_detrended_quad = phases - np.polyval(coeffs_quad, times_centered)
        # Linear — diagnostic.
        coeffs_lin = np.polyfit(times_centered, phases, deg=1)
        phase_detrended_lin = phases - np.polyval(coeffs_lin, times_centered)
    except (np.linalg.LinAlgError, ValueError):
        # Degenerate input (would need n_samples < 3 — already excluded
        # by min_samples ≥ 10).  Fall back to mean-subtraction; both
        # linear and quadratic collapse to the same residual.
        mode_doppler_hz = 0.0
        phase_detrended_quad = phases - float(np.mean(phases))
        phase_detrended_lin = phase_detrended_quad

    sigma_phi_quadratic_rad = float(np.std(phase_detrended_quad, ddof=0))
    sigma_phi_linear_rad = float(np.std(phase_detrended_lin, ddof=0))
    # Canonical σ_φ value used for severity classification — matches
    # the v0.5.2 behaviour (quadratic).
    sigma_phi_rad = sigma_phi_quadratic_rad

    # Underfit ratio: ≥ 1 by construction (quadratic basis ⊇ linear
    # basis → quadratic residual ≤ linear residual).  When the
    # quadratic residual is exactly zero (pathological clean signal),
    # the ratio is 1.0 by convention rather than ∞.
    if sigma_phi_quadratic_rad > 0.0:
        sigma_phi_underfit_ratio = (
            sigma_phi_linear_rad / sigma_phi_quadratic_rad
        )
    else:
        sigma_phi_underfit_ratio = 1.0

    # ─── NaN/Inf guard ───────────────────────────────────────────────
    if not (np.isfinite(s4_index) and np.isfinite(sigma_phi_rad)
            and np.isfinite(mode_doppler_hz)
            and np.isfinite(sigma_phi_linear_rad)
            and np.isfinite(sigma_phi_underfit_ratio)):
        return _unknown_result(n_input, n_outliers_rejected)

    # ─── Confidence ──────────────────────────────────────────────────
    # Pure sample-count saturation on the *retained* count; no CV-based
    # suppression (that would inversely correlate with S4 — see module
    # docstring).
    confidence = float(min(1.0, n_samples / CONFIDENCE_SAMPLE_SATURATION))

    s4_severity = _s4_severity(s4_index)
    sigma_phi_severity = _sigma_phi_severity(sigma_phi_rad)
    scintillation_event = (
        s4_index >= S4_EVENT_THRESHOLD
        or sigma_phi_rad >= SIGMA_PHI_EVENT_THRESHOLD
    )

    return ScintillationResult(
        s4_index=s4_index,
        s4_severity=s4_severity,
        sigma_phi_rad=sigma_phi_rad,
        sigma_phi_severity=sigma_phi_severity,
        scintillation_event=scintillation_event,
        confidence=confidence,
        n_samples=n_samples,
        n_outliers_rejected=n_outliers_rejected,
        mode_doppler_hz=mode_doppler_hz,
        sigma_phi_linear_rad=sigma_phi_linear_rad,
        sigma_phi_quadratic_rad=sigma_phi_quadratic_rad,
        sigma_phi_underfit_ratio=sigma_phi_underfit_ratio,
    )

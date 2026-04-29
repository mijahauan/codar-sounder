"""Trace extraction — find ionospheric peaks in a range profile.

A dechirped CPI yields a power-vs-range profile.  At the lowest
ranges (group range ≈ ground distance) we see strong direct ground-wave
energy; at longer ranges we see sky-wave reflections (E-region,
F-region, sporadic-E).  We want the F-region peak — the one that maps
to a useful virtual height via Kaeppler Eq. 10.

v0.1 strategy:

1.  Maintain a slowly-varying ground-clutter mask by median-filtering
    the last N profiles.  Subtracting that out removes time-stable
    structure (direct path, persistent backscatter) and emphasises
    transient sky-wave returns.

2.  Within an operator-configured ``[range_min_km, range_max_km]``
    window, find the strongest residual peak.

3.  Compute SNR as the peak-to-median ratio (in dB) within the search
    window.  If SNR is below the configured threshold, return ``None``
    (no usable detection this CPI).

This is deliberately simple — automatic multi-mode classification
(E/F/Es) and high/low-ray separation are deferred to a future release.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TraceDetection:
    """One peak detection from one CPI's range profile."""
    group_range_km: float
    snr_db: float
    power: float                 # raw power at the peak (units arbitrary)
    bin_index: int               # index into the (positive-range) profile


class GroundClutterMask:
    """Slowly-adapting median mask of previous range profiles.

    Maintains the last ``window`` profiles in a deque and serves the
    pointwise median as a clutter estimate.  The median is robust to
    transient sky-wave peaks — the F-region returns we *want* to keep
    move enough between CPIs that they don't dominate the median, while
    direct-path / ground-clutter energy is stationary and gets
    subtracted out cleanly.

    Subtraction is clamped to zero so a profile with sub-clutter
    excursions doesn't produce negative power values.
    """

    def __init__(self, window: int = 20):
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        self.window = window
        self._profiles: deque[np.ndarray] = deque(maxlen=window)

    def update(self, profile: np.ndarray) -> None:
        """Add a profile to the rolling window."""
        self._profiles.append(np.asarray(profile, dtype=np.float32))

    @property
    def n_observations(self) -> int:
        return len(self._profiles)

    def estimate(self, length: int) -> np.ndarray:
        """Return the current clutter estimate (median of buffered profiles).

        If no profiles have been observed yet, returns a zero vector of
        the requested length.
        """
        if not self._profiles:
            return np.zeros(length, dtype=np.float32)
        stack = np.stack(self._profiles, axis=0)
        return np.median(stack, axis=0).astype(np.float32)

    def subtract(self, profile: np.ndarray) -> np.ndarray:
        """Return ``max(profile - clutter, 0)``."""
        clutter = self.estimate(len(profile))
        return np.clip(profile - clutter, 0.0, None)


def _power_to_db(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, eps))


def find_f_region_peak(
    profile: np.ndarray,
    range_axis_km: np.ndarray,
    *,
    range_min_km: float,
    range_max_km: float,
    snr_threshold_db: float,
    clutter_mask: Optional[GroundClutterMask] = None,
) -> Optional[TraceDetection]:
    """Find the strongest F-region peak in a range profile.

    Args:
        profile: 1-D power-vs-range vector (sorted by ascending range,
            positive-range half only — see ``positive_range_window``).
        range_axis_km: companion 1-D array of group-range values (km).
        range_min_km / range_max_km: search-window bounds.
        snr_threshold_db: minimum (peak / median-in-window) ratio in dB.
        clutter_mask: optional rolling-median ground-clutter estimator.
            If provided, the mask is updated with the current profile
            and the residual is searched.

    Returns ``None`` if the search window is empty, the window has fewer
    than three samples (peak detection nonsense), or the peak fails the
    SNR threshold.
    """
    if profile.shape != range_axis_km.shape:
        raise ValueError(
            f"profile shape {profile.shape} != range_axis shape "
            f"{range_axis_km.shape}"
        )
    if range_min_km >= range_max_km:
        raise ValueError(
            f"range_min_km ({range_min_km}) must be < range_max_km "
            f"({range_max_km})"
        )

    if clutter_mask is not None:
        residual = clutter_mask.subtract(profile)
        clutter_mask.update(profile)
    else:
        residual = profile

    window_mask = (range_axis_km >= range_min_km) & (range_axis_km <= range_max_km)
    if not np.any(window_mask):
        return None

    window_indices = np.where(window_mask)[0]
    if window_indices.size < 3:
        return None

    window_powers = residual[window_indices]
    rel_peak = int(np.argmax(window_powers))
    peak_idx = int(window_indices[rel_peak])
    peak_power = float(residual[peak_idx])

    # SNR is the peak in dB above the median of the search window.
    # Median is robust to a few strong samples either side of the peak.
    # Cap at 60 dB to avoid runaway values when the clutter mask has
    # zeroed the median (which produces log10(eps) ≈ -300 → bogus
    # 300+ dB SNRs).  60 dB is well above any plausible real SNR.
    median_power = float(np.median(window_powers))
    snr_db = min(60.0, float(
        _power_to_db(np.array([peak_power]))[0]
        - _power_to_db(np.array([median_power]))[0]
    ))

    if snr_db < snr_threshold_db:
        return None

    return TraceDetection(
        group_range_km=float(range_axis_km[peak_idx]),
        snr_db=snr_db,
        power=peak_power,
        bin_index=peak_idx,
    )

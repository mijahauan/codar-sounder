"""Geometric inversion of group range to virtual height and equivalent
vertical frequency.

Implements Kaeppler et al. (2022, *Atmos. Meas. Tech.* 15:4531-4545)
Eq. 10 (mirror model) and Eq. 11 (secant law), plus their uncertainty
propagation Eq. 13/14, generalised for multi-hop returns (v0.7).

The model treats the ionosphere as an idealised reflecting mirror at
altitude ``h`` (the *virtual height*) above the ground midpoint of an
oblique path of ground-distance ``D`` between transmitter and receiver.
For ``N`` hops, the wave traces ``N`` symmetric isosceles paths, each
spanning ground distance ``D/N`` and group range ``P/N``::

       (P/(2N))² = h² + (D/(2N))²        (multi-hop Eq. 10)

so::

       h = sqrt(P² - D²) / (2N)

For ``N = 1`` this is Kaeppler's original Eq. 10.  v0.5/0.6 always
assumed ``N = 1``; v0.7 selects ``N`` via :func:`select_n_hops` based
on which hop count gives the most climatologically plausible virtual
height.

The secant law and takeoff zenith angle are *independent* of ``N``
(the per-hop and total geometry have the same ratio ``D/P``), so::

       sin(φ) = D / P              (geometry-invariant)
       fv     = fo · cos(φ) = fo · sqrt(P² - D²) / P    (Eq. 11)

Why multi-hop matters
---------------------
Live verification on bee1-rx888 SEAB (1416 km path, 2026-05-21)
found 35% of detections classified as F2_extreme (h'(N=1) > 500 km)
even on geomagnetically quiet days — a rate that's physically
implausible for genuine F2_extreme conditions.  The
``tasks/analysis/2026-05-21_f2_extreme_multihop_diagnostic.md``
report showed 100% of those records have a clean 3-hop interpretation
at typical F2 heights (mean h' = 263 km), strongly supporting the
multi-hop misclassification hypothesis.  v0.7's selection logic
reinterprets apparent-F2_extreme records as the smallest ``N ≥ 2``
giving a plausible F2 height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Speed of light, km/s.  Kaeppler uses the same constant; consistency
# matters for cross-validation against their published Zenodo dataset.
C_KM_PER_S = 299_792.458


@dataclass(frozen=True)
class IonosphericFix:
    """One geometric inversion of a single group-range measurement.

    ``virtual_height_km`` and ``mode_layer`` reflect the ``n_hops``-
    corrected interpretation (v0.7).  Records produced by v0.5/0.6
    always assumed ``n_hops = 1``; downstream consumers reading mixed
    archives should consult ``processing_version`` to know which
    convention applies.
    """
    group_range_km: float
    ground_distance_km: float
    virtual_height_km: float
    virtual_height_uncertainty_km: float
    equivalent_vertical_freq_mhz: float
    equivalent_vertical_freq_uncertainty_mhz: float
    takeoff_zenith_deg: float
    mode_layer: str              # "E", "F1", "F2", "F2_extreme", "below_E", or "unknown"
    n_hops: int = 1              # 1, 2, 3, or 4 — selected by select_n_hops()


# Virtual-height boundaries (km) used by classify_layer().  These match
# the conventional ionospheric-layer altitudes in Davies, *Ionospheric
# Radio* (1990) and the AIPL/CCIR profiles used by digisondes:
#
#   * E layer       : 90  – 140 km  (also covers most sporadic-E)
#   * F1 layer      : 140 – 220 km  (daytime; merges into F2 at night)
#   * F2 layer      : 220 – 500 km  (always present)
#   * F2 extreme    : > 500 km      (atypical; often nighttime F-bulge
#                                   or a 2F2 multi-hop misclassified as
#                                   single-hop, kept distinct so an
#                                   analyst can flag it)
#   * below E       : < 90  km      (very unusual; usually ground
#                                   clutter that escaped the mask, or
#                                   a non-physical mirror-model artefact)
#
# Sporadic-E (Es) cannot be reliably distinguished from regular E
# without a frequency-vs-MUF context (digisondes do this with multiple
# sweeps).  We collapse both to "E" and let downstream consumers
# disambiguate from the diurnal/seasonal pattern.
_LAYER_BOUNDARIES_KM = {
    "below_E":     (None,  90.0),
    "E":           ( 90.0, 140.0),
    "F1":          (140.0, 220.0),
    "F2":          (220.0, 500.0),
    "F2_extreme":  (500.0,  None),
}


# Threshold above which apparent-1-hop virtual height is considered
# suspicious (v0.7) — matches the F2_extreme classification boundary.
# Selections only attempt multi-hop reinterpretation when h_1 ≥ this.
_MULTIHOP_TRIGGER_H_KM = 500.0

# Multi-hop search bounds: the corrected h_N must fall in this band to
# qualify as a "plausible F-region reflection."  Spans F1 + F2
# (Davies 1990 conventions).
_PLAUSIBLE_F_LOW_KM = 150.0
_PLAUSIBLE_F_HIGH_KM = 500.0

# Maximum hop count to consider — N ≥ 5 would imply path loss > ~60 dB
# typical at HF, well below codar-sounder's typical SNR threshold.
DEFAULT_MAX_HOPS = 4


def classify_layer(virtual_height_km: float) -> str:
    """Map a virtual height (km) to a coarse ionospheric layer label.

    Returns one of: ``"E"``, ``"F1"``, ``"F2"``, ``"F2_extreme"``,
    ``"below_E"``.  See :data:`_LAYER_BOUNDARIES_KM` for the
    altitude-band rationale.

    Returns ``"unknown"`` only for non-finite input (NaN / inf), which
    can arise from a degenerate group-range / ground-distance ratio.
    """
    import math
    if not math.isfinite(virtual_height_km):
        return "unknown"
    for label, (lo, hi) in _LAYER_BOUNDARIES_KM.items():
        if (lo is None or virtual_height_km >= lo) and (
            hi is None or virtual_height_km < hi
        ):
            return label
    return "unknown"


def virtual_height_km(
    group_range_km: float,
    ground_distance_km: float,
    n_hops: int = 1,
) -> float:
    """Mirror-model virtual height from group range and TX-RX distance.

    Generalised multi-hop form (Kaeppler 2022 Eq. 10 extended)::

        h = sqrt(P² - D²) / (2N)

    For ``N = 1`` (default) this reduces to Kaeppler's original
    expression.  Larger ``N`` describes a signal that has bounced
    between the ionosphere and ground ``N`` times — typical for
    long-path returns where the apparent 1-hop interpretation puts
    the reflection at implausibly high altitudes.

    Args:
        group_range_km: total group-path length P (km).
        ground_distance_km: great-circle distance D between TX and RX (km).
        n_hops: number of ionospheric reflections (1, 2, 3, ...).

    Raises:
        ValueError: if ``P <= D`` (the mirror model is geometrically
            unphysical — the wave would have to travel a path shorter
            than the ground distance, i.e. through the ground), or
            ``n_hops < 1``.
    """
    if n_hops < 1:
        raise ValueError(f"n_hops must be >= 1; got {n_hops}")
    if group_range_km <= ground_distance_km:
        raise ValueError(
            f"group_range_km ({group_range_km}) must exceed "
            f"ground_distance_km ({ground_distance_km}) for the "
            f"mirror model to admit a real solution"
        )
    return math.sqrt(group_range_km ** 2 - ground_distance_km ** 2) / (2.0 * n_hops)


def equivalent_vertical_freq_mhz(
    oblique_freq_mhz: float,
    group_range_km: float,
    ground_distance_km: float,
) -> float:
    """Secant-law equivalent vertical frequency at the path midpoint.

    Kaeppler 2022 Eq. 11 (rearranged)::

        fv = fo · cos(φ) = fo · sqrt(P² - D²) / P

    where ``φ`` is the takeoff zenith angle from the sub-ionospheric foot.

    Args:
        oblique_freq_mhz: RF frequency of the oblique-path observation (MHz).
        group_range_km: total group-path length P (km).
        ground_distance_km: great-circle distance D between TX and RX (km).
    """
    if group_range_km <= ground_distance_km:
        raise ValueError(
            f"group_range_km ({group_range_km}) must exceed "
            f"ground_distance_km ({ground_distance_km}) for the "
            f"secant law to admit a real solution"
        )
    cos_phi = math.sqrt(group_range_km ** 2 - ground_distance_km ** 2) / group_range_km
    return oblique_freq_mhz * cos_phi


def takeoff_zenith_deg(group_range_km: float, ground_distance_km: float) -> float:
    """Takeoff zenith angle ``φ`` (degrees) from sin(φ) = D/P."""
    if group_range_km <= 0:
        raise ValueError(f"group_range_km must be > 0; got {group_range_km}")
    sin_phi = min(1.0, ground_distance_km / group_range_km)
    return math.degrees(math.asin(sin_phi))


def virtual_height_uncertainty_km(
    group_range_km: float,
    ground_distance_km: float,
    group_range_uncertainty_km: float,
    ground_distance_uncertainty_km: float = 0.0,
    n_hops: int = 1,
) -> float:
    """Propagated uncertainty in virtual height (Kaeppler Eq. 13,
    multi-hop generalised).

    For ``N`` hops::

        h  = sqrt(P² - D²) / (2N)
        dh/dP = P / (4N²·h)

    Solving for ``Δh`` (ΔD = 0)::

        Δh = ΔP · P / (4·N²·h)

    For N = 1 this reduces to the original Kaeppler expression.  The
    ``N²`` denominator captures that a multi-hop reflection involves
    ``N`` independent geometric "samples" of the same layer, each
    contributing ``ΔP/N`` to the per-hop uncertainty.
    """
    if n_hops < 1:
        raise ValueError(f"n_hops must be >= 1; got {n_hops}")
    h = virtual_height_km(group_range_km, ground_distance_km, n_hops=n_hops)
    if h == 0:
        return 0.0
    inner = (
        group_range_uncertainty_km ** 2
        - (ground_distance_km / group_range_km) ** 2
        * ground_distance_uncertainty_km ** 2
    )
    if inner < 0:
        # ΔD overpowers ΔP — model breakdown; report ΔP-derived bound.
        inner = group_range_uncertainty_km ** 2
    return (group_range_km / (4.0 * n_hops ** 2 * h)) * math.sqrt(inner)


def equivalent_vertical_freq_uncertainty_mhz(
    oblique_freq_mhz: float,
    group_range_km: float,
    ground_distance_km: float,
    group_range_uncertainty_km: float,
) -> float:
    """Propagated uncertainty in equivalent vertical frequency
    (Kaeppler Eq. 14)::

        Δfv = fo · D² · ΔP / (P² · sqrt(P² - D²))
    """
    if group_range_km <= ground_distance_km:
        raise ValueError("group_range_km must exceed ground_distance_km")
    return (
        oblique_freq_mhz
        * ground_distance_km ** 2
        * group_range_uncertainty_km
        / (group_range_km ** 2 * math.sqrt(group_range_km ** 2 - ground_distance_km ** 2))
    )


def select_n_hops(
    group_range_km: float,
    ground_distance_km: float,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> int:
    """Pick the most plausible hop count for an observed (P, D).

    Strategy (v0.7):
      1. Compute ``h_1 = virtual_height_km(P, D, 1)``.
      2. If ``h_1 < _MULTIHOP_TRIGGER_H_KM`` (typical E / F1 / F2
         heights), return 1 — the 1-hop interpretation is the
         simplest and historically the default.  Preserves legacy
         behaviour for all genuine 1-hop returns.
      3. Otherwise (apparent F2_extreme), search ``N = 2..max_hops``
         for the smallest N giving ``h_N`` in the
         ``[_PLAUSIBLE_F_LOW_KM, _PLAUSIBLE_F_HIGH_KM]`` band.
      4. If no N qualifies (e.g. extreme group_range / D ratios),
         fall back to ``N = 1`` so the measurement is reported under
         legacy conventions and flagged as F2_extreme.

    A "smallest plausible N" tie-breaker favours the higher-SNR
    interpretation: 2-hop returns are typically stronger than 3-hop
    on the same path, so when both 2-hop and 3-hop give valid F
    heights, 2-hop is the safer pick.  The selection is *necessarily
    ambiguous* without polarization or AOA — the
    ``tasks/analysis/2026-05-21_f2_extreme_multihop_diagnostic.md``
    report covers when both N = 2 and N = 3 simultaneously qualify.
    """
    if group_range_km <= ground_distance_km:
        raise ValueError(
            f"group_range_km ({group_range_km}) must exceed "
            f"ground_distance_km ({ground_distance_km})"
        )
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1; got {max_hops}")
    h_1 = math.sqrt(group_range_km ** 2 - ground_distance_km ** 2) / 2.0
    if h_1 < _MULTIHOP_TRIGGER_H_KM:
        return 1
    for n in range(2, max_hops + 1):
        h_n = h_1 / n
        if _PLAUSIBLE_F_LOW_KM <= h_n <= _PLAUSIBLE_F_HIGH_KM:
            return n
    return 1


def invert(
    group_range_km: float,
    ground_distance_km: float,
    oblique_freq_mhz: float,
    group_range_uncertainty_km: float = 0.0,
    n_hops: int | None = None,
) -> IonosphericFix:
    """Combined inversion: returns one :class:`IonosphericFix` per
    measurement.

    Args:
        group_range_km: total group-path length P (km).
        ground_distance_km: TX-RX great-circle distance D (km).
        oblique_freq_mhz: RF frequency at which the return was observed.
        group_range_uncertainty_km: 1σ uncertainty in P (km).
        n_hops: optional override for the hop count.  When ``None``
            (default), :func:`select_n_hops` chooses the most
            plausible N from the geometry.  Explicit values are
            useful for testing or for callers that have external
            disambiguation (e.g. polarization, AOA).
    """
    if n_hops is None:
        n_hops = select_n_hops(group_range_km, ground_distance_km)
    h = virtual_height_km(group_range_km, ground_distance_km, n_hops=n_hops)
    # fv and takeoff zenith are N-invariant — same expression for any N.
    fv = equivalent_vertical_freq_mhz(
        oblique_freq_mhz, group_range_km, ground_distance_km
    )
    dh = virtual_height_uncertainty_km(
        group_range_km, ground_distance_km, group_range_uncertainty_km,
        n_hops=n_hops,
    )
    dfv = equivalent_vertical_freq_uncertainty_mhz(
        oblique_freq_mhz, group_range_km, ground_distance_km, group_range_uncertainty_km
    )
    phi = takeoff_zenith_deg(group_range_km, ground_distance_km)
    return IonosphericFix(
        group_range_km=group_range_km,
        ground_distance_km=ground_distance_km,
        virtual_height_km=h,
        virtual_height_uncertainty_km=dh,
        equivalent_vertical_freq_mhz=fv,
        equivalent_vertical_freq_uncertainty_mhz=dfv,
        takeoff_zenith_deg=phi,
        mode_layer=classify_layer(h),
        n_hops=n_hops,
    )


def group_range_resolution_km(sweep_bandwidth_hz: float) -> float:
    """One-way group-range resolution of an FMCW sounder.

    For bistatic sky-wave propagation the range resolution equals the
    inverse-bandwidth time delay times c::

        ΔP = c / BW

    For ``BW = 25.7 kHz`` (the 4.5 MHz CODAR band) this yields ~11.7 km
    — matching Kaeppler's quoted ±12 km uncertainty.
    """
    if sweep_bandwidth_hz <= 0:
        raise ValueError(f"sweep_bandwidth_hz must be > 0; got {sweep_bandwidth_hz}")
    return C_KM_PER_S / sweep_bandwidth_hz

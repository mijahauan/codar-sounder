"""Geometric inversion of group range to virtual height and equivalent
vertical frequency.

Implements Kaeppler et al. (2022, *Atmos. Meas. Tech.* 15:4531-4545)
Eq. 10 (mirror model) and Eq. 11 (secant law), plus their uncertainty
propagation Eq. 13/14.

The model treats the ionosphere as an idealised reflecting mirror at
altitude ``h`` (the *virtual height*) above the ground midpoint of an
oblique path of ground-distance ``D`` between transmitter and receiver.
A radio wave that travels from TX at the surface up to the mirror and
down to RX traces a group path of total length ``P``::

       (P/2)² = h² + (D/2)²        (Eq. 10)

so::

       h = sqrt(P² - D²) / 2

The secant law converts the oblique-path frequency at which a radar
return is observed (``fo``) to the equivalent vertical-incidence
frequency at the path midpoint (``fv``) using the path's takeoff
zenith angle ``φ`` from the sub-ionospheric foot of the wave::

       sin(φ) = D / P              (geometry of the same triangle)
       fv     = fo · cos(φ) = fo · sqrt(P² - D²) / P    (Eq. 11)

These two transforms together let a CODAR-sounder reduce a measured
group-range time series at a fixed RF frequency to a (virtual_height,
equivalent_vertical_frequency) time series at the path midpoint —
directly comparable to a vertical-incidence digisonde reading there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Speed of light, km/s.  Kaeppler uses the same constant; consistency
# matters for cross-validation against their published Zenodo dataset.
C_KM_PER_S = 299_792.458


@dataclass(frozen=True)
class IonosphericFix:
    """One geometric inversion of a single group-range measurement."""
    group_range_km: float
    ground_distance_km: float
    virtual_height_km: float
    virtual_height_uncertainty_km: float
    equivalent_vertical_freq_mhz: float
    equivalent_vertical_freq_uncertainty_mhz: float
    takeoff_zenith_deg: float


def virtual_height_km(group_range_km: float, ground_distance_km: float) -> float:
    """Mirror-model virtual height from group range and TX-RX distance.

    Kaeppler 2022 Eq. 10::

        h = sqrt(P² - D²) / 2

    Args:
        group_range_km: total group-path length P (km).
        ground_distance_km: great-circle distance D between TX and RX (km).

    Raises:
        ValueError: if ``P <= D`` (the mirror model is geometrically
            unphysical — the wave would have to travel a path shorter
            than the ground distance, i.e. through the ground).
    """
    if group_range_km <= ground_distance_km:
        raise ValueError(
            f"group_range_km ({group_range_km}) must exceed "
            f"ground_distance_km ({ground_distance_km}) for the "
            f"mirror model to admit a real solution"
        )
    return math.sqrt(group_range_km ** 2 - ground_distance_km ** 2) / 2.0


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
) -> float:
    """Propagated uncertainty in virtual height (Kaeppler Eq. 13).

    Solving Kaeppler's Eq. 13 for ``Δh`` (with ``ΔP``, ``ΔD`` known)::

        ΔP² = (4h/P)² Δh² + (D/P)² ΔD²
        Δh  = (P / 4h) · sqrt(ΔP² − (D/P)² ΔD²)

    For ``ΔD = 0`` (TX/RX coordinates known precisely, the typical
    case)::

        Δh = ΔP · P / (4h)
    """
    h = virtual_height_km(group_range_km, ground_distance_km)
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
    return (group_range_km / (4.0 * h)) * math.sqrt(inner)


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


def invert(
    group_range_km: float,
    ground_distance_km: float,
    oblique_freq_mhz: float,
    group_range_uncertainty_km: float = 0.0,
) -> IonosphericFix:
    """Combined inversion: returns one ``IonosphericFix`` per measurement."""
    h = virtual_height_km(group_range_km, ground_distance_km)
    fv = equivalent_vertical_freq_mhz(
        oblique_freq_mhz, group_range_km, ground_distance_km
    )
    dh = virtual_height_uncertainty_km(
        group_range_km, ground_distance_km, group_range_uncertainty_km
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

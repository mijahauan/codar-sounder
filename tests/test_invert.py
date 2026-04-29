"""Tests for core/invert.py — Kaeppler 2022 Eq. 10/11/13/14."""

from __future__ import annotations

import math

import pytest

from codar_sounder.core.invert import (
    C_KM_PER_S,
    IonosphericFix,
    equivalent_vertical_freq_mhz,
    equivalent_vertical_freq_uncertainty_mhz,
    group_range_resolution_km,
    invert,
    takeoff_zenith_deg,
    virtual_height_km,
    virtual_height_uncertainty_km,
)


# ---------------------------------------------------------------------------
# Mirror model (Eq. 10)
# ---------------------------------------------------------------------------

class TestVirtualHeight:

    def test_isosceles_triangle(self):
        """A path with P=1000, D=600 has virtual height sqrt(P^2-D^2)/2 = 400."""
        h = virtual_height_km(group_range_km=1000.0, ground_distance_km=600.0)
        assert math.isclose(h, 400.0)

    def test_pythagorean_triple(self):
        """3-4-5 triple: D=6, P=10 → h=4."""
        h = virtual_height_km(10.0, 6.0)
        assert math.isclose(h, 4.0)

    def test_known_kaeppler_geometry(self):
        """A typical Kaeppler-paper case: DUCK→CARL ~664 km ground distance,
        F-region group range ~700 km → virtual height ~111 km."""
        h = virtual_height_km(700.0, 664.0)
        # sqrt(700^2 - 664^2) / 2 = sqrt(49104) / 2 ≈ 110.79
        assert 110 < h < 112

    def test_raises_when_path_shorter_than_ground(self):
        with pytest.raises(ValueError, match="must exceed"):
            virtual_height_km(500.0, 600.0)

    def test_raises_when_path_equals_ground(self):
        with pytest.raises(ValueError):
            virtual_height_km(600.0, 600.0)


# ---------------------------------------------------------------------------
# Secant law (Eq. 11)
# ---------------------------------------------------------------------------

class TestEquivalentVerticalFreq:

    def test_vertical_incidence_returns_oblique_freq(self):
        """When D=0 (vertical incidence), fv = fo."""
        # Strictly D=0 violates 'P>D' guard; use a tiny D instead
        fv = equivalent_vertical_freq_mhz(
            oblique_freq_mhz=4.5, group_range_km=600.0, ground_distance_km=0.001
        )
        assert math.isclose(fv, 4.5, rel_tol=1e-6)

    def test_secant_law_kaeppler_geometry(self):
        """4.537 MHz oblique freq, P=700, D=600 → fv = 4.537 * sqrt(700²-600²)/700.

        sqrt(490000 - 360000) = sqrt(130000) ≈ 360.555
        fv ≈ 4.537 * 360.555 / 700 ≈ 2.337 MHz
        """
        fv = equivalent_vertical_freq_mhz(4.537, 700.0, 600.0)
        assert math.isclose(fv, 2.337, abs_tol=0.005)

    def test_secant_law_at_grazing(self):
        """For very oblique paths (D → P), fv → 0."""
        fv = equivalent_vertical_freq_mhz(
            oblique_freq_mhz=4.537,
            group_range_km=1000.0,
            ground_distance_km=999.99,
        )
        assert fv < 0.1


# ---------------------------------------------------------------------------
# Takeoff zenith
# ---------------------------------------------------------------------------

class TestTakeoffZenith:

    def test_vertical_incidence(self):
        phi = takeoff_zenith_deg(group_range_km=600.0, ground_distance_km=0.0)
        assert math.isclose(phi, 0.0)

    def test_grazing_incidence(self):
        phi = takeoff_zenith_deg(group_range_km=600.0, ground_distance_km=600.0)
        assert math.isclose(phi, 90.0)

    def test_45_degrees(self):
        # sin(phi) = D/P = 0.5 → phi = 30°
        phi = takeoff_zenith_deg(group_range_km=600.0, ground_distance_km=300.0)
        assert math.isclose(phi, 30.0)


# ---------------------------------------------------------------------------
# Uncertainty propagation (Eq. 13/14)
# ---------------------------------------------------------------------------

class TestVirtualHeightUncertainty:

    def test_zero_uncertainty(self):
        dh = virtual_height_uncertainty_km(700.0, 600.0, 0.0)
        assert math.isclose(dh, 0.0)

    def test_known_proportionality(self):
        """ΔD=0 case: Δh = ΔP * P / (4h).
        For P=700, D=600, h=190.787, ΔP=12: Δh = 12 * 700 / (4 * 190.787) ≈ 11.0
        """
        dh = virtual_height_uncertainty_km(700.0, 600.0, 12.0)
        h = virtual_height_km(700.0, 600.0)
        expected = 12.0 * 700.0 / (4.0 * h)
        assert math.isclose(dh, expected, rel_tol=1e-9)

    def test_kaeppler_typical_case(self):
        """Kaeppler's typical reported uncertainty: ~12 km group-range
        resolution, near-vertical paths give ~8 km virtual-height uncertainty."""
        # P=700, D=600, ΔP=12 → Δh ≈ 11
        # P=400, D=300, ΔP=12 → Δh ≈ 9 (closer to vertical, better resolution)
        dh = virtual_height_uncertainty_km(400.0, 300.0, 12.0)
        assert 6 < dh < 12


class TestEquivalentVerticalFreqUncertainty:

    def test_zero_uncertainty(self):
        dfv = equivalent_vertical_freq_uncertainty_mhz(4.537, 700.0, 600.0, 0.0)
        assert math.isclose(dfv, 0.0)

    def test_scales_with_group_range_uncertainty(self):
        d1 = equivalent_vertical_freq_uncertainty_mhz(4.537, 700.0, 600.0, 12.0)
        d2 = equivalent_vertical_freq_uncertainty_mhz(4.537, 700.0, 600.0, 24.0)
        assert math.isclose(d2, 2.0 * d1, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Combined invert()
# ---------------------------------------------------------------------------

class TestInvert:

    def test_returns_ionospheric_fix(self):
        fix = invert(
            group_range_km=700.0,
            ground_distance_km=664.0,
            oblique_freq_mhz=4.537,
            group_range_uncertainty_km=11.7,
        )
        assert isinstance(fix, IonosphericFix)
        assert fix.group_range_km == 700.0
        assert fix.ground_distance_km == 664.0
        assert 100 < fix.virtual_height_km < 120
        assert fix.virtual_height_uncertainty_km > 0
        assert fix.equivalent_vertical_freq_mhz < fix.group_range_km
        assert 0 < fix.takeoff_zenith_deg < 90

    def test_self_consistent(self):
        """Recompute h from fv and check it matches."""
        fix = invert(900.0, 700.0, 4.537, group_range_uncertainty_km=12.0)
        # fv = fo * sqrt(P^2 - D^2) / P
        # sqrt(P^2 - D^2) / P = fv / fo
        # sqrt(P^2 - D^2) = P * fv / fo
        # h = sqrt(P^2 - D^2) / 2 = P * fv / (2 * fo)
        h_from_fv = fix.group_range_km * fix.equivalent_vertical_freq_mhz / (
            2.0 * 4.537
        )
        assert math.isclose(h_from_fv, fix.virtual_height_km, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Range resolution
# ---------------------------------------------------------------------------

class TestGroupRangeResolution:

    def test_kaeppler_figure(self):
        """25.7 kHz CODAR sweep bandwidth → ~11.7 km group-range resolution
        (Kaeppler 2022 reports ±12 km)."""
        res = group_range_resolution_km(25_734.0)
        assert 11 < res < 12

    def test_higher_bandwidth_better_resolution(self):
        # 50 kHz CODAR (~13 MHz band)
        assert group_range_resolution_km(50_000.0) < group_range_resolution_km(25_000.0)

    def test_zero_or_negative_bandwidth_raises(self):
        with pytest.raises(ValueError):
            group_range_resolution_km(0)
        with pytest.raises(ValueError):
            group_range_resolution_km(-1.0)

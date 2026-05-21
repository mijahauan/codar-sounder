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
    select_n_hops,
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


# ---------------------------------------------------------------------------
# Multi-hop generalisation (v0.7)
# ---------------------------------------------------------------------------

class TestVirtualHeightMultiHop:
    """``h = sqrt(P²-D²) / (2N)`` — divides by 2N instead of 2."""

    def test_n_hops_1_matches_default(self):
        """The N=1 case must equal the historical 1-hop expression."""
        h_default = virtual_height_km(1000.0, 600.0)
        h_n1 = virtual_height_km(1000.0, 600.0, n_hops=1)
        assert math.isclose(h_default, h_n1)
        assert math.isclose(h_default, 400.0)

    def test_n_hops_2_halves_apparent_h(self):
        """For the same (P, D), N=2 gives half the height of N=1."""
        h_1 = virtual_height_km(2000.0, 1416.0, n_hops=1)
        h_2 = virtual_height_km(2000.0, 1416.0, n_hops=2)
        assert math.isclose(h_2, h_1 / 2.0)

    def test_n_hops_3_thirds(self):
        h_1 = virtual_height_km(2400.0, 1416.0, n_hops=1)
        h_3 = virtual_height_km(2400.0, 1416.0, n_hops=3)
        assert math.isclose(h_3, h_1 / 3.0)

    def test_seab_2064_3hop_lands_in_normal_F(self):
        """The diagnostic example: SEAB at D=1416, P=2064 → h_3 ≈ 250 km."""
        h_3 = virtual_height_km(2064.0, 1416.0, n_hops=3)
        assert 240 < h_3 < 260

    def test_invalid_n_hops_raises(self):
        with pytest.raises(ValueError, match="n_hops"):
            virtual_height_km(1000.0, 600.0, n_hops=0)
        with pytest.raises(ValueError, match="n_hops"):
            virtual_height_km(1000.0, 600.0, n_hops=-1)


class TestVirtualHeightUncertaintyMultiHop:
    """Multi-hop uncertainty: Δh = ΔP·P / (4·N²·h).  N² in denominator,
    so multi-hop returns have proportionally lower height uncertainty."""

    def test_n_hops_1_matches_default(self):
        dh_default = virtual_height_uncertainty_km(700.0, 600.0, 12.0)
        dh_n1 = virtual_height_uncertainty_km(700.0, 600.0, 12.0, n_hops=1)
        assert math.isclose(dh_default, dh_n1)

    def test_n_squared_scaling(self):
        """At the same (P, D, ΔP), uncertainty scales as 1/N²."""
        # Use a long-range case so h_N > 0 for N up to 3.
        P, D, dP = 2400.0, 1416.0, 12.0
        dh_1 = virtual_height_uncertainty_km(P, D, dP, n_hops=1)
        dh_3 = virtual_height_uncertainty_km(P, D, dP, n_hops=3)
        # Δh_N = ΔP·P / (4·N²·h_N).  With h_N = h_1/N, this gives
        # Δh_N = ΔP·P / (4·N²·h_1/N) = ΔP·P / (4·N·h_1) = Δh_1 / N.
        # So dh_3 = dh_1 / 3.
        assert math.isclose(dh_3, dh_1 / 3.0, rel_tol=1e-9)


class TestSelectNHops:
    """``select_n_hops`` returns 1 for typical 1-hop returns and ≥ 2
    when an F2_extreme apparent-h has a plausible multi-hop
    interpretation."""

    def test_typical_F2_returns_n1(self):
        """A clean F2 return at h_1 = 300 km should pick N=1."""
        # P, D for h_1 = 300: P = 2·sqrt(D²/4 + 300²) = sqrt(D² + 360000)
        D = 1000.0
        P = math.sqrt(D * D + 4 * 300 * 300)
        assert select_n_hops(P, D) == 1

    def test_F1_returns_n1(self):
        """F1-region returns at h_1 ≈ 180 km pick N=1."""
        D = 700.0
        P = math.sqrt(D * D + 4 * 180 * 180)
        assert select_n_hops(P, D) == 1

    def test_E_returns_n1(self):
        """E-region returns at h_1 ≈ 110 km pick N=1."""
        D = 500.0
        P = math.sqrt(D * D + 4 * 110 * 110)
        assert select_n_hops(P, D) == 1

    def test_seab_2064_picks_2hop(self):
        """SEAB at P=2064, D=1416: h_1 = 751 km (F2_extreme), h_2 =
        375 km (normal F2) — selector picks N=2 (smallest N with
        plausible F-region h)."""
        n = select_n_hops(2064.0, 1416.0)
        assert n == 2

    def test_seab_2300_picks_2hop_when_2hop_plausible(self):
        """At P=2300, D=1416: h_1 = 904 km, h_2 = 452 km — within
        F2 band (≤ 500), so N=2 wins.  3-hop also plausible (h_3 =
        301 km) but selector prefers smallest N."""
        n = select_n_hops(2300.0, 1416.0)
        assert n == 2

    def test_seab_2800_picks_3hop_when_2hop_too_high(self):
        """At very long P, the 2-hop interpretation can be too high
        for F-region — selector picks the next N down.  P=2800,
        D=1416: h_1 = 1206 km, h_2 = 603 km (out of F band),
        h_3 = 402 km (in F band) → N=3."""
        n = select_n_hops(2800.0, 1416.0)
        assert n == 3

    def test_fallback_to_n1_when_no_plausible_interpretation(self):
        """Synthetic extreme: P just barely above D, h_1 ~ 0 — picks
        N=1 (the trigger height isn't reached so multi-hop isn't
        considered)."""
        n = select_n_hops(1001.0, 1000.0)
        assert n == 1

    def test_geometry_violation_raises(self):
        with pytest.raises(ValueError, match="must exceed"):
            select_n_hops(500.0, 600.0)


class TestInvertMultiHop:
    """End-to-end: ``invert()`` auto-selects N and reports it on the fix."""

    def test_typical_1hop_fix_reports_n_hops_1(self):
        fix = invert(
            group_range_km=700.0,
            ground_distance_km=664.0,
            oblique_freq_mhz=4.537,
            group_range_uncertainty_km=11.7,
        )
        assert fix.n_hops == 1
        # h ≈ 110.8 km unchanged from v0.6 behaviour at this geometry.
        assert 100 < fix.virtual_height_km < 120

    def test_seab_F2_extreme_apparent_becomes_F2_at_n_2(self):
        """SEAB-shaped input: P=2064 km, D=1416 km.  Under v0.5/0.6
        this was h=751 km mode_layer=F2_extreme.  Under v0.7 it's
        h=375 km mode_layer=F2 with n_hops=2."""
        fix = invert(
            group_range_km=2064.0,
            ground_distance_km=1416.0,
            oblique_freq_mhz=13.45,
            group_range_uncertainty_km=6.0,
        )
        assert fix.n_hops == 2
        assert 350 < fix.virtual_height_km < 400
        assert fix.mode_layer == "F2"

    def test_explicit_n_hops_override(self):
        """Callers can force a specific N (testing / external
        disambiguation)."""
        fix_auto = invert(2064.0, 1416.0, 13.45, n_hops=1)
        assert fix_auto.n_hops == 1
        # Forced N=1 → original v0.5/0.6 behaviour: F2_extreme.
        assert fix_auto.virtual_height_km > 600
        assert fix_auto.mode_layer == "F2_extreme"

    def test_fv_unchanged_across_n_hops(self):
        """The equivalent vertical frequency is N-invariant — same
        geometry, different n_hops, same fv."""
        fix_1 = invert(2064.0, 1416.0, 13.45, n_hops=1)
        fix_2 = invert(2064.0, 1416.0, 13.45, n_hops=2)
        fix_3 = invert(2064.0, 1416.0, 13.45, n_hops=3)
        assert math.isclose(
            fix_1.equivalent_vertical_freq_mhz,
            fix_2.equivalent_vertical_freq_mhz, rel_tol=1e-12,
        )
        assert math.isclose(
            fix_1.equivalent_vertical_freq_mhz,
            fix_3.equivalent_vertical_freq_mhz, rel_tol=1e-12,
        )

    def test_takeoff_zenith_unchanged_across_n_hops(self):
        """sin(φ) = D/P, also N-invariant."""
        fix_1 = invert(2064.0, 1416.0, 13.45, n_hops=1)
        fix_2 = invert(2064.0, 1416.0, 13.45, n_hops=2)
        assert math.isclose(
            fix_1.takeoff_zenith_deg, fix_2.takeoff_zenith_deg,
            rel_tol=1e-12,
        )

    def test_h_uncertainty_smaller_for_multihop(self):
        """Δh = ΔP·P / (4·N²·h_N) shrinks proportionally with N."""
        fix_1 = invert(2064.0, 1416.0, 13.45, group_range_uncertainty_km=12.0, n_hops=1)
        fix_2 = invert(2064.0, 1416.0, 13.45, group_range_uncertainty_km=12.0, n_hops=2)
        assert fix_2.virtual_height_uncertainty_km < fix_1.virtual_height_uncertainty_km

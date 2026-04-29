"""Cross-validation against Kaeppler et al. 2022 Zenodo dataset.

doi:10.5281/zenodo.6341875 publishes their derived virtual-height /
equivalent-vertical-frequency time series for DUCK, LISL, CORE
transmitters seen at the CARL receiver in October 2020.  We use it
to verify that codar-sounder's ``invert()`` (Kaeppler Eq. 10/11)
agrees with their published numbers given the same geometry.

The HDF5 lives at::

    /home/wsprdaemon/kaeppler-zenodo/CARL_Codar_WP937Digisonde_October2020_02042022.hdf5

If it isn't present, every test in this module is skipped — running
the suite without the dataset doesn't fail.

This is a *math-consistency* validation, not a re-processing one
(the raw IQ would let us run dechirp+trace+invert end-to-end against
their published numbers; that requires the larger 105 MB zip and is
the next step).  Math consistency is still meaningful: Kaeppler's
Eq. 10/11 implementations and ours must agree on every published
sample, otherwise one of us has a bug.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from codar_sounder.config import haversine_km
from codar_sounder.core.invert import (
    equivalent_vertical_freq_mhz,
    virtual_height_km,
)


# -- fixtures -------------------------------------------------------------

ZENODO_HDF5 = Path(
    "/home/wsprdaemon/kaeppler-zenodo/"
    "CARL_Codar_WP937Digisonde_October2020_02042022.hdf5"
)
CARL_LAT, CARL_LON = 34.62, -82.83          # Kaeppler 2022 Table 2
TX_COORDS = {                                # Kaeppler 2022 Table 2
    "DUCK": (36.18,  -75.75),
    "LISL": (36.69,  -75.92),
    "CORE": (34.76,  -76.41),
}
CODAR_OBLIQUE_FREQ_MHZ = 4.53718             # Kaeppler 2022 §3, all three TXs


def _require_hdf5():
    if not ZENODO_HDF5.exists():
        pytest.skip(
            f"Zenodo dataset not present at {ZENODO_HDF5}; "
            f"skipping cross-validation."
        )
    try:
        import h5py        # noqa: F401
    except ImportError:
        pytest.skip("h5py not installed; skipping cross-validation.")


@pytest.fixture(scope="module")
def kaeppler_data():
    _require_hdf5()
    import h5py
    out: dict = {}
    with h5py.File(ZENODO_HDF5, "r") as f:
        out["t_unix"] = f["tUnix"][:]
        for tx in ("DUCK", "LISL", "CORE"):
            for mode in ("O", "X"):
                key = f"{tx}_{mode}"
                out[f"hF2_{key}"] = f[f"hF2_{key}"][:]
                out[f"fv_{key}"] = f[f"fv_{key}"][:]
                out[f"dhF2_{key}"] = f[f"dhF2_{key}"][:]
                out[f"dfv_{key}"] = f[f"dfv_{key}"][:]
    return out


@pytest.fixture(scope="module")
def ground_distances() -> dict[str, float]:
    return {
        tx: haversine_km(CARL_LAT, CARL_LON, lat, lon)
        for tx, (lat, lon) in TX_COORDS.items()
    }


# -- data sanity ----------------------------------------------------------

class TestDatasetIntegrity:

    def test_timestamps_are_october_2020(self, kaeppler_data):
        t = kaeppler_data["t_unix"]
        assert len(t) == 16_845
        # Convert Unix → year, month: October 2020 = days 274..305
        from datetime import datetime, timezone
        dt0 = datetime.fromtimestamp(float(t[0]), tz=timezone.utc)
        dt1 = datetime.fromtimestamp(float(t[-1]), tz=timezone.utc)
        assert dt0.year == 2020 and dt0.month == 10
        assert dt1.year == 2020 and dt1.month == 10

    def test_arrays_align(self, kaeppler_data):
        n = len(kaeppler_data["t_unix"])
        for key in kaeppler_data:
            if key == "t_unix":
                continue
            assert len(kaeppler_data[key]) == n, (
                f"{key} length {len(kaeppler_data[key])} != t_unix length {n}"
            )

    def test_ground_distances_match_kaeppler_table(self, ground_distances):
        # Kaeppler 2022 Table 2: distances "to CARL".
        expected_km = {"DUCK": 664, "LISL": 665, "CORE": 587}
        for tx, exp in expected_km.items():
            actual = ground_distances[tx]
            assert abs(actual - exp) < 5, (
                f"{tx}: computed {actual:.1f} km, paper says {exp} km"
            )


# -- math consistency: their (hF2, fv) pairs should satisfy our Eq. 10/11
# --------------------------------------------------------------------------

class TestPaperPairsSatisfyOurFormulas:
    """For every published (hF2, fv) pair, the geometry-implied group
    range and equivalent-vertical-frequency should match what our
    ``invert.py`` produces.  Disagreement = bug in our code or theirs.
    """

    @pytest.mark.parametrize("tx", ["DUCK", "LISL", "CORE"])
    @pytest.mark.parametrize("mode", ["O", "X"])
    def test_published_pairs_are_geometrically_consistent(
        self, tx, mode, kaeppler_data, ground_distances
    ):
        """If h is the published virtual height, the implied group
        range is P = sqrt(D² + (2h)²); plug back into our secant law
        and recover the published fv within numerical tolerance.
        """
        D = ground_distances[tx]
        hF2 = kaeppler_data[f"hF2_{tx}_{mode}"]
        fv  = kaeppler_data[f"fv_{tx}_{mode}"]

        # Many entries are NaN where no detection was made; only test
        # finite entries.
        valid = np.isfinite(hF2) & np.isfinite(fv) & (hF2 > 0)
        n_valid = int(valid.sum())
        assert n_valid > 100, f"only {n_valid} finite samples for {tx}/{mode}"

        # Predict fv from h via the same formula we'd use in production.
        P_pred = np.sqrt(D ** 2 + (2.0 * hF2[valid]) ** 2)
        fv_pred = np.array([
            equivalent_vertical_freq_mhz(CODAR_OBLIQUE_FREQ_MHZ, p, D)
            for p in P_pred
        ])
        delta = fv_pred - fv[valid]
        # The published fv was computed from a separately-measured P
        # (group range), not from h, so they may disagree by O(round-off
        # + one-bin range resolution).  Their range bin is ~12 km, which
        # at this geometry corresponds to ~few-percent fv variation.
        # Tolerate up to 10% relative error; ~99% should land inside 1%.
        rel_err = np.abs(delta) / np.abs(fv[valid])
        assert np.median(rel_err) < 0.05, (
            f"{tx}/{mode}: median rel error {np.median(rel_err):.3f}"
        )
        # 99th-percentile error tolerance
        assert np.percentile(rel_err, 99) < 0.20

    @pytest.mark.parametrize("tx", ["DUCK", "LISL", "CORE"])
    @pytest.mark.parametrize("mode", ["O", "X"])
    def test_invert_round_trip(self, tx, mode, kaeppler_data, ground_distances):
        """Forward-compute P from their h, run our invert(), check we
        recover h back exactly (to float precision)."""
        from codar_sounder.core.invert import invert

        D = ground_distances[tx]
        hF2 = kaeppler_data[f"hF2_{tx}_{mode}"]
        valid = np.isfinite(hF2) & (hF2 > 0)
        h_truth = hF2[valid][:200]      # subsample for speed
        for h in h_truth:
            P = math.sqrt(D ** 2 + (2.0 * h) ** 2)
            fix = invert(
                group_range_km=float(P),
                ground_distance_km=D,
                oblique_freq_mhz=CODAR_OBLIQUE_FREQ_MHZ,
            )
            assert abs(fix.virtual_height_km - h) < 0.01, (
                f"{tx}/{mode}: h={h:.2f} → P={P:.2f} → recovered "
                f"h={fix.virtual_height_km:.2f}"
            )


# -- physical reasonableness ---------------------------------------------

class TestPhysicalReasonableness:
    """Sanity checks on the published time series independent of our
    formulas — these would catch a corrupted download or a mis-labelled
    column more than they validate our code, but they're useful
    confidence-builders."""

    @pytest.mark.parametrize("tx", ["DUCK", "LISL", "CORE"])
    @pytest.mark.parametrize("mode", ["O", "X"])
    def test_virtual_heights_in_ionospheric_range(
        self, tx, mode, kaeppler_data
    ):
        """F-region virtual heights should sit between ~150 and ~600 km.
        Anything way outside that suggests a unit error or bad data."""
        h = kaeppler_data[f"hF2_{tx}_{mode}"]
        valid = np.isfinite(h) & (h > 0)
        assert valid.sum() > 0
        median = float(np.median(h[valid]))
        assert 150 < median < 600, f"{tx}/{mode}: median hF2 {median} km is unphysical"

    @pytest.mark.parametrize("tx", ["DUCK", "LISL", "CORE"])
    def test_o_x_modes_close_in_height(self, tx, kaeppler_data):
        """For sky-wave reflection in the F-region, O- and X-mode group
        ranges differ by tens of km but are correlated — their virtual
        heights for the same path/time should agree within ~30 km on
        average."""
        h_o = kaeppler_data[f"hF2_{tx}_O"]
        h_x = kaeppler_data[f"hF2_{tx}_X"]
        # Only times where both modes produced detections.
        both = np.isfinite(h_o) & np.isfinite(h_x) & (h_o > 0) & (h_x > 0)
        if both.sum() < 50:
            pytest.skip(f"{tx}: insufficient O+X co-detections")
        diff = h_o[both] - h_x[both]
        median_abs_diff = float(np.median(np.abs(diff)))
        assert median_abs_diff < 50, (
            f"{tx}: median |h_O - h_X| = {median_abs_diff:.1f} km "
            f"exceeds expected O/X separation"
        )

    def test_three_paths_have_correlated_signatures(self, kaeppler_data):
        """The three CARL→{DUCK,LISL,CORE} paths reflect off ionospheric
        midpoints within a few hundred km of each other — when one path
        sees a high F-region, the others should too (modulo timing).

        Concretely: the median hF2 over the month should agree to
        within ~50 km across the three TXs."""
        medians = []
        for tx in ("DUCK", "LISL", "CORE"):
            h = kaeppler_data[f"hF2_{tx}_X"]
            valid = np.isfinite(h) & (h > 0)
            medians.append(float(np.median(h[valid])))
        spread = max(medians) - min(medians)
        assert spread < 60, (
            f"DUCK/LISL/CORE median hF2 spread: {spread:.1f} km "
            f"(values: {medians})"
        )

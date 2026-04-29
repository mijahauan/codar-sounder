"""Tests for core/tdma.py — co-band TX offset discovery.

Each test synthesises a CPI containing two TDMA-multiplexed
transmitters, runs ``discover_tx_offsets``, and asserts the discovered
peak positions match the synthesis truth within a tolerance set by the
cross-correlation main-lobe width.
"""

from __future__ import annotations

import numpy as np
import pytest

from codar_sounder.core.dechirp import C_KM_PER_S
from codar_sounder.core.tdma import (
    TDMAPeak,
    discover_tx_offsets,
    match_peaks_to_txs,
    offset_for_tx,
)

# Match test_dechirp.py for consistency.
SAMPLE_RATE_HZ = 64000.0
SWEEP_RATE_HZ_PER_S = -25733.913
SRF_HZ = 1.0
N_SAMPLES = int(SAMPLE_RATE_HZ / SRF_HZ)


def _two_tx_tdma_iq(
    tdma_a_s: float,
    tdma_b_s: float,
    delay_a_s: float,
    delay_b_s: float,
    n_periods: int = 4,
    amp_a: float = 1.0,
    amp_b: float = 1.0,
    noise_db: float = -80.0,
) -> np.ndarray:
    """Synthesise n periods of IQ with two TDMA-distinct CODAR TXs."""
    n_total = n_periods * N_SAMPLES
    t = np.arange(n_total) / SAMPLE_RATE_HZ
    rx = np.zeros(n_total, dtype=np.complex64)
    for tdma, delay, amp in (
        (tdma_a_s, delay_a_s, amp_a),
        (tdma_b_s, delay_b_s, amp_b),
    ):
        t_into_sweep = (t - tdma - delay) % (1.0 / SRF_HZ)
        valid = t >= (tdma + delay)
        phase = 2.0 * np.pi * 0.5 * SWEEP_RATE_HZ_PER_S * t_into_sweep ** 2
        echo = amp * np.exp(1j * phase).astype(np.complex64)
        echo[~valid] = 0
        rx += echo
    if noise_db > -200:
        sigma = 10 ** (noise_db / 20.0)
        rng = np.random.default_rng(seed=42)
        rx += sigma * (
            rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)
        ).astype(np.complex64)
    return rx


class TestDiscoverPeaks:

    def test_finds_two_distinct_peaks(self):
        """Two TXs with distinct TDMA offsets and distinct ranges should
        produce two distinct cross-correlation peaks.
        """
        # TX_A: TDMA offset 0, ground delay 600 km / c
        # TX_B: TDMA offset 0.4 s, ground delay 1200 km / c
        delay_a = 600.0 / C_KM_PER_S
        delay_b = 1200.0 / C_KM_PER_S
        tdma_a, tdma_b = 0.0, 0.4

        rx = _two_tx_tdma_iq(
            tdma_a_s=tdma_a, tdma_b_s=tdma_b,
            delay_a_s=delay_a, delay_b_s=delay_b,
        )

        peaks = discover_tx_offsets(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
            snr_threshold_db=10.0,
        )
        assert len(peaks) >= 2, f"expected ≥2 peaks, got {len(peaks)}"

        # Expected peak positions: TX's correlation peak appears at
        # (tdma_offset + ground_delay) in samples.
        expected_a = int(round((tdma_a + delay_a) * SAMPLE_RATE_HZ))
        expected_b = int(round((tdma_b + delay_b) * SAMPLE_RATE_HZ))

        # Both expected positions should be within 4 samples of one of
        # the discovered peaks.
        offsets = [p.offset_samples for p in peaks]

        def closest_dist(target: int) -> int:
            return min(
                min(abs(o - target), N_SAMPLES - abs(o - target))
                for o in offsets
            )

        assert closest_dist(expected_a) <= 4, (
            f"TX_A expected near sample {expected_a}; "
            f"discovered offsets {offsets}"
        )
        assert closest_dist(expected_b) <= 4, (
            f"TX_B expected near sample {expected_b}; "
            f"discovered offsets {offsets}"
        )

    def test_peaks_sorted_by_snr_desc(self):
        rx = _two_tx_tdma_iq(
            tdma_a_s=0.0, tdma_b_s=0.4,
            delay_a_s=600.0 / C_KM_PER_S,
            delay_b_s=1200.0 / C_KM_PER_S,
            amp_a=1.0,
            amp_b=0.3,                  # weaker → lower SNR
        )
        peaks = discover_tx_offsets(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
            snr_threshold_db=5.0,
        )
        assert len(peaks) >= 2
        snrs = [p.snr_db for p in peaks]
        assert snrs == sorted(snrs, reverse=True)

    def test_min_separation_collapses_distant_peaks_when_huge(self):
        """A min_separation larger than the period collapses everything
        to one peak — the global maximum.

        Two TXs cleanly separated in the period, but we ask for an
        impossibly-wide min_separation; only the strongest survives.
        """
        rx = _two_tx_tdma_iq(
            tdma_a_s=0.0, tdma_b_s=0.4,
            delay_a_s=600.0 / C_KM_PER_S,
            delay_b_s=1200.0 / C_KM_PER_S,
        )
        peaks = discover_tx_offsets(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
            snr_threshold_db=10.0,
            min_separation_samples=N_SAMPLES // 2 + 1,    # > half period
        )
        assert len(peaks) == 1

    def test_no_signal_returns_empty(self):
        """Pure noise → no peaks above SNR threshold."""
        n_total = 4 * N_SAMPLES
        rng = np.random.default_rng(seed=1)
        rx = (
            rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)
        ).astype(np.complex64)
        peaks = discover_tx_offsets(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
            snr_threshold_db=15.0,
        )
        assert peaks == []

    def test_short_input_raises(self):
        rx = np.zeros(100, dtype=np.complex64)
        with pytest.raises(ValueError):
            discover_tx_offsets(
                rx,
                sample_rate_hz=SAMPLE_RATE_HZ,
                sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
                sweep_repetition_hz=SRF_HZ,
            )

    def test_one_period_input_raises(self):
        """Multi-period folding requires ≥2 periods of rx."""
        rx = np.zeros(N_SAMPLES, dtype=np.complex64)
        with pytest.raises(ValueError, match="two sweep periods"):
            discover_tx_offsets(
                rx,
                sample_rate_hz=SAMPLE_RATE_HZ,
                sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
                sweep_repetition_hz=SRF_HZ,
            )

    def test_real_input_rejected(self):
        rx = np.zeros(N_SAMPLES, dtype=np.float32)
        with pytest.raises(ValueError, match="must be complex"):
            discover_tx_offsets(
                rx,
                sample_rate_hz=SAMPLE_RATE_HZ,
                sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
                sweep_repetition_hz=SRF_HZ,
            )


class TestOffsetForTX:

    def test_subtracts_direct_delay(self):
        """A correlation peak at sample = (tdma + ground_delay) * Fs
        should yield offset = tdma * Fs after we subtract ground_delay.
        """
        D_km = 600.0
        tdma_offset_seconds = 0.25
        direct_delay_s = D_km / C_KM_PER_S
        peak_pos = int(round((tdma_offset_seconds + direct_delay_s) * SAMPLE_RATE_HZ))

        offset = offset_for_tx(
            peak_pos,
            ground_distance_km=D_km,
            sample_rate_hz=SAMPLE_RATE_HZ,
            n_per_sweep=N_SAMPLES,
        )
        expected = int(round(tdma_offset_seconds * SAMPLE_RATE_HZ))
        # Allow one sample of rounding slack (direct_delay_samples is rounded).
        assert abs(offset - expected) <= 1

    def test_wraps_when_subtraction_underflows(self):
        """If ground delay > correlation peak position, the modulo wraps cleanly."""
        offset = offset_for_tx(
            correlation_peak_samples=10,
            ground_distance_km=1500.0,            # 320 samples direct delay
            sample_rate_hz=SAMPLE_RATE_HZ,
            n_per_sweep=N_SAMPLES,
        )
        assert 0 <= offset < N_SAMPLES


class TestMatchPeaksToTXs:

    def test_assigns_strongest_peak_to_closest_tx(self):
        """LISL/ASSA/CEDR-shaped scenario: three TXs at increasing distances,
        three discovered peaks of decreasing SNR — closest TX gets the
        loudest peak.
        """
        peaks = [
            TDMAPeak(offset_samples=100, snr_db=30.0, correlation_power=1e6),
            TDMAPeak(offset_samples=20000, snr_db=25.0, correlation_power=3e5),
            TDMAPeak(offset_samples=40000, snr_db=20.0, correlation_power=1e5),
        ]
        tx_distances = {"LISL": 380.0, "ASSA": 560.0, "CEDR": 580.0}

        out = match_peaks_to_txs(
            peaks, tx_distances,
            sample_rate_hz=SAMPLE_RATE_HZ, n_per_sweep=N_SAMPLES,
        )
        # Each TX should have an integer offset (not None).
        assert all(v is not None for v in out.values())
        # LISL is closest → gets the strongest peak (offset 100 - direct_delay).
        # ASSA and CEDR are nearly equidistant → next two peaks.
        # We just verify all three got distinct offsets.
        offsets = list(out.values())
        assert len(set(offsets)) == 3

    def test_returns_none_when_no_peaks(self):
        out = match_peaks_to_txs(
            [], {"LISL": 380.0},
            sample_rate_hz=SAMPLE_RATE_HZ, n_per_sweep=N_SAMPLES,
        )
        assert out == {"LISL": None}

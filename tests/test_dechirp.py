"""Tests for core/dechirp.py — synthetic FMCW IQ → expected range bin.

The signal-processing core is the part of codar-sounder that's most
dependent on getting the math right (and least observable from a
high-level smoke test), so the tests here are deliberately
ground-truth: synthesise a chirp at a known target delay, run it
through dechirp(), and assert the peak appears in the expected range
bin within one bin's tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from codar_sounder.core.dechirp import (
    C_KM_PER_S,
    DechirpResult,
    dechirp,
    make_replica,
    positive_range_window,
    range_profile,
)


# Standard test parameters: 4.5 MHz CODAR-like sweep at a manageable
# sample rate.  64000 Hz is enough to comfortably span the 25.7 kHz
# CODAR sweep BW with ~2× oversampling, while keeping the FFT small.
SAMPLE_RATE_HZ = 64000.0
SWEEP_RATE_HZ_PER_S = -25733.913       # CODAR 4.537 MHz down-chirp
SRF_HZ = 1.0                            # 1 sweep per second
N_SAMPLES = int(SAMPLE_RATE_HZ / SRF_HZ)


def _synth_chirp(
    n_total_samples: int,
    target_delays_s: list[float],
    target_amplitudes: list[float],
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    sweep_rate_hz_per_s: float = SWEEP_RATE_HZ_PER_S,
    srf_hz: float = SRF_HZ,
    noise_db: float = -60.0,
    target_tdma_offsets_s: list[float] | None = None,
) -> np.ndarray:
    """Synthesise a CPI of received IQ for a given list of targets.

    Each ``target`` contributes a delayed copy of the transmitted chirp
    at the specified delay (seconds) and amplitude (linear).  Targets'
    chirps wrap into the next sweep period naturally — the modulo
    on ``t`` reproduces a continuously transmitting CODAR.

    ``target_tdma_offsets_s`` (per-target, defaults to all zero) shifts
    each TX's sweep-start time within the period — used to synthesise
    multiple co-band TDMA-multiplexed transmitters.  An offset of 0.5 s
    on a 1 Hz SRF means the TX's chirp is half-way through its period
    when our buffer starts.
    """
    if target_tdma_offsets_s is None:
        target_tdma_offsets_s = [0.0] * len(target_delays_s)
    if len(target_tdma_offsets_s) != len(target_delays_s):
        raise ValueError("target_tdma_offsets_s must match target_delays_s in length")

    t = np.arange(n_total_samples) / sample_rate_hz
    sweep_period = 1.0 / srf_hz

    rx = np.zeros(n_total_samples, dtype=np.complex64)
    for delay, amp, tdma in zip(
        target_delays_s, target_amplitudes, target_tdma_offsets_s
    ):
        # The TX's sweep starts at t = tdma; its chirp at our receiver
        # arrives delayed by the propagation time `delay`.  At sample n
        # (time t = n/Fs) the TX has been sweeping for
        # ((t - tdma - delay) mod sweep_period) seconds.
        t_into_sweep = (t - tdma - delay) % sweep_period
        valid = t >= (tdma + delay)
        phase = 2.0 * np.pi * 0.5 * sweep_rate_hz_per_s * t_into_sweep ** 2
        echo = amp * np.exp(1j * phase).astype(np.complex64)
        echo[~valid] = 0
        rx += echo

    if noise_db > -200:
        # Additive complex Gaussian noise.  A target with amplitude 1.0
        # has unit power; noise at -60 dB = 1e-6 power = 1e-3 amplitude.
        noise_amplitude = 10 ** (noise_db / 20.0)
        rng = np.random.default_rng(seed=42)
        rx += noise_amplitude * (
            rng.standard_normal(n_total_samples)
            + 1j * rng.standard_normal(n_total_samples)
        ).astype(np.complex64)

    return rx


# ---------------------------------------------------------------------------
# make_replica
# ---------------------------------------------------------------------------

class TestMakeReplica:

    def test_length(self):
        r = make_replica(N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S)
        assert r.shape == (N_SAMPLES,)

    def test_complex_dtype(self):
        r = make_replica(N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S)
        assert r.dtype.kind == "c"

    def test_unwindowed_unit_magnitude(self):
        r = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S, window=False
        )
        assert np.allclose(np.abs(r), 1.0, atol=1e-6)

    def test_windowed_attenuates_edges(self):
        """Hann window: edge samples should be near zero, centre near unity."""
        r = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S, window=True
        )
        assert abs(r[0]) < 0.01
        assert abs(r[-1]) < 0.01
        assert 0.95 < abs(r[N_SAMPLES // 2]) <= 1.0

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            make_replica(0, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S)
        with pytest.raises(ValueError):
            make_replica(N_SAMPLES, 0, SWEEP_RATE_HZ_PER_S)


# ---------------------------------------------------------------------------
# dechirp() — single-target ground-truth recovery
# ---------------------------------------------------------------------------

class TestDechirpSingleTarget:

    def _run(self, target_range_km: float, n_sweeps: int = 4):
        """Synthesise a single-target CPI and dechirp it; return profile."""
        target_delay_s = target_range_km / C_KM_PER_S
        rx = _synth_chirp(
            n_total_samples=n_sweeps * N_SAMPLES,
            target_delays_s=[target_delay_s],
            target_amplitudes=[1.0],
        )
        result = dechirp(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
        )
        prof = range_profile(result)
        ranges, prof_pos = positive_range_window(result, prof)
        return result, ranges, prof_pos

    @pytest.mark.parametrize("target_range_km", [200.0, 500.0, 1000.0, 2000.0])
    def test_recovers_known_range(self, target_range_km):
        """Peak must land within ~12 km (one Kaeppler resolution cell) of truth."""
        _, ranges, prof = self._run(target_range_km)
        peak_idx = int(np.argmax(prof))
        peak_range_km = float(ranges[peak_idx])
        # Use 2x the Kaeppler resolution as tolerance — synthetic
        # signals with windowed replicas spread peaks slightly.
        assert abs(peak_range_km - target_range_km) < 25.0, (
            f"target {target_range_km} km, peak at {peak_range_km} km"
        )

    def test_higher_amplitude_target_dominates(self):
        """Two targets at different ranges; the louder one wins the peak."""
        rx = _synth_chirp(
            n_total_samples=4 * N_SAMPLES,
            target_delays_s=[
                500.0 / C_KM_PER_S,    # weak
                1500.0 / C_KM_PER_S,   # 10x stronger
            ],
            target_amplitudes=[0.1, 1.0],
        )
        result = dechirp(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
        )
        ranges, prof = positive_range_window(result, range_profile(result))
        peak_idx = int(np.argmax(prof))
        assert abs(ranges[peak_idx] - 1500.0) < 25.0


class TestDechirpMultipath:

    def test_two_distinct_targets_both_visible(self):
        """When two strong targets are well-separated in range, both
        should appear as separate peaks in the range profile.
        """
        rx = _synth_chirp(
            n_total_samples=8 * N_SAMPLES,
            target_delays_s=[
                400.0 / C_KM_PER_S,
                1200.0 / C_KM_PER_S,
            ],
            target_amplitudes=[1.0, 1.0],
            noise_db=-80.0,
        )
        result = dechirp(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
        )
        ranges, prof = positive_range_window(result, range_profile(result))
        # Find local maxima above 50% of global max
        threshold = 0.5 * prof.max()
        peaks_km = []
        for i in range(1, len(prof) - 1):
            if prof[i] > threshold and prof[i] >= prof[i-1] and prof[i] >= prof[i+1]:
                peaks_km.append(float(ranges[i]))
        # Both peaks must be among the detected maxima.
        assert any(abs(p - 400.0) < 25.0 for p in peaks_km), \
            f"target at 400 km not detected; peaks: {peaks_km}"
        assert any(abs(p - 1200.0) < 25.0 for p in peaks_km), \
            f"target at 1200 km not detected; peaks: {peaks_km}"


# ---------------------------------------------------------------------------
# DechirpResult / output shape
# ---------------------------------------------------------------------------

class TestDechirpOutputShape:

    def test_returns_dechirp_result(self):
        rx = _synth_chirp(
            n_total_samples=4 * N_SAMPLES,
            target_delays_s=[500.0 / C_KM_PER_S],
            target_amplitudes=[1.0],
        )
        result = dechirp(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
        )
        assert isinstance(result, DechirpResult)
        assert result.range_doppler.shape == (4, N_SAMPLES)
        assert result.range_axis_km.shape == (N_SAMPLES,)
        assert result.doppler_axis_hz.shape == (4,)

    def test_short_input_raises(self):
        rx = np.zeros(100, dtype=np.complex64)
        with pytest.raises(ValueError):
            dechirp(
                rx,
                sample_rate_hz=SAMPLE_RATE_HZ,
                sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
                sweep_repetition_hz=SRF_HZ,
            )

    def test_real_input_rejected(self):
        rx = np.zeros(N_SAMPLES * 4, dtype=np.float32)
        with pytest.raises(ValueError, match="must be complex"):
            dechirp(
                rx,
                sample_rate_hz=SAMPLE_RATE_HZ,
                sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
                sweep_repetition_hz=SRF_HZ,
            )

    def test_zero_srf_rejected(self):
        rx = np.zeros(N_SAMPLES * 4, dtype=np.complex64)
        with pytest.raises(ValueError):
            dechirp(
                rx,
                sample_rate_hz=SAMPLE_RATE_HZ,
                sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
                sweep_repetition_hz=0,
            )


# ---------------------------------------------------------------------------
# TDMA — phase-offset replica separates co-band transmitters
# ---------------------------------------------------------------------------

class TestPhaseOffsetReplica:
    """A replica with `phase_offset_samples=k` is the same chirp
    sweeping for `k/Fs` seconds before our buffer's sample 0 — its
    instantaneous frequency at t=0 is κ·k/Fs Hz, not 0.  Verifies the
    new parameter wires through correctly without breaking v0.2 paths.
    """

    def test_zero_offset_matches_v02_default(self):
        r0 = make_replica(N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S)
        r0_default = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S,
            phase_offset_samples=0,
        )
        assert np.array_equal(r0, r0_default)

    def test_offset_changes_replica(self):
        r0 = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S, window=False,
        )
        r_off = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S, window=False,
            phase_offset_samples=N_SAMPLES // 4,
        )
        # Offsetting should not produce a trivially-equal replica.
        assert not np.allclose(r0, r_off)

    def test_full_period_offset_wraps_to_zero(self):
        """Offset = N is one full period → wraps modulo N → identical to 0."""
        r0 = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S, window=False,
        )
        r_full = make_replica(
            N_SAMPLES, SAMPLE_RATE_HZ, SWEEP_RATE_HZ_PER_S, window=False,
            phase_offset_samples=N_SAMPLES,
        )
        assert np.allclose(r0, r_full, atol=1e-5)


class TestTDMASeparation:
    """Two TXs at different TDMA offsets in the same band should each be
    cleanly extracted by their own offset replica, with substantial
    cross-suppression of the other.
    """

    def _two_tx_iq(self, tdma_a_s: float, tdma_b_s: float, n_sweeps: int = 8):
        """Synthesise a CPI containing TX_A at one delay and TX_B at another,
        each with a distinct TDMA sweep-start offset.
        """
        # Two distinct propagation delays so we can tell which TX
        # produced which peak in the dechirp output.
        delay_a = 600.0 / C_KM_PER_S         # TX_A: 600 km direct
        delay_b = 1200.0 / C_KM_PER_S        # TX_B: 1200 km direct
        return _synth_chirp(
            n_total_samples=n_sweeps * N_SAMPLES,
            target_delays_s=[delay_a, delay_b],
            target_amplitudes=[1.0, 1.0],
            target_tdma_offsets_s=[tdma_a_s, tdma_b_s],
            noise_db=-80.0,
        )

    def test_offset_a_extracts_tx_a_suppresses_tx_b(self):
        """Dechirping with TX_A's offset → strong peak at TX_A's range,
        weak/diffuse energy at TX_B's range.
        """
        # TX_A at offset 0, TX_B at offset 0.5 s (half-period).
        rx = self._two_tx_iq(tdma_a_s=0.0, tdma_b_s=0.5)

        # Replica aligned to TX_A (zero offset).
        result_a = dechirp(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
            phase_offset_samples=0,
        )
        ranges, prof_a = positive_range_window(result_a, range_profile(result_a))

        # Find peak power near TX_A's true range (600 km) and near TX_B's (1200 km).
        def power_near(range_km: float, tol_km: float = 30.0) -> float:
            mask = np.abs(ranges - range_km) < tol_km
            return float(prof_a[mask].max()) if np.any(mask) else 0.0

        p_at_a = power_near(600.0)
        p_at_b = power_near(1200.0)

        # When we dechirp with TX_A's offset, TX_A should dominate.
        # Cross-suppression: ≥10 dB stronger at TX_A than at TX_B.
        ratio_db = 10.0 * np.log10(p_at_a / max(p_at_b, 1e-30))
        assert ratio_db > 10.0, (
            f"TX_A peak {10*np.log10(p_at_a):.1f} dB vs TX_B "
            f"{10*np.log10(p_at_b):.1f} dB; expected ≥10 dB suppression"
        )

    def test_offset_b_extracts_tx_b_suppresses_tx_a(self):
        """Symmetric: dechirping with TX_B's offset → TX_B wins."""
        rx = self._two_tx_iq(tdma_a_s=0.0, tdma_b_s=0.5)

        # TX_B's TDMA offset = 0.5 s = N_SAMPLES // 2 samples.
        offset_b_samples = int(0.5 * SAMPLE_RATE_HZ)
        result_b = dechirp(
            rx,
            sample_rate_hz=SAMPLE_RATE_HZ,
            sweep_rate_hz_per_s=SWEEP_RATE_HZ_PER_S,
            sweep_repetition_hz=SRF_HZ,
            phase_offset_samples=offset_b_samples,
        )
        ranges, prof_b = positive_range_window(result_b, range_profile(result_b))

        def power_near(range_km: float, tol_km: float = 30.0) -> float:
            mask = np.abs(ranges - range_km) < tol_km
            return float(prof_b[mask].max()) if np.any(mask) else 0.0

        p_at_a = power_near(600.0)
        p_at_b = power_near(1200.0)

        ratio_db = 10.0 * np.log10(p_at_b / max(p_at_a, 1e-30))
        assert ratio_db > 10.0, (
            f"TX_B peak {10*np.log10(p_at_b):.1f} dB vs TX_A "
            f"{10*np.log10(p_at_a):.1f} dB; expected ≥10 dB suppression"
        )

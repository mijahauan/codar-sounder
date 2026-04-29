"""FMCW dechirping engine — Kaeppler 2022 §2.1.

A coastal CODAR transmitter radiates a linear-frequency-modulated
continuous-wave (FMCW) signal: the carrier sweeps linearly across a
narrow band (e.g. 25 kHz at 4.5 MHz) once per sweep-repetition period
(typically 1 s), with a known signed sweep rate ``κ`` (Hz/s).  At a
distant receiver, multiple delayed copies of that chirp arrive — direct
ground-wave plus one or more sky-wave (ionospherically-reflected)
modes.

Dechirping recovers each propagation mode's group-range delay by
mixing the received signal with a phase-coherent replica of the
transmitted chirp.  The mixer output is a "beat" frequency
proportional to the path delay::

       f_beat = κ · τ           ⇒    τ = f_beat / κ
       group_range = c · τ

So a 100 Hz beat tone at κ = -25.7 kHz/s indicates a target at
group-range ≈ c · (100 / 25700) s ≈ 1167 km.

This module implements Kaeppler's frequency-domain formulation:
  1. Build a windowed replica of one sweep.
  2. Reshape received IQ into an M × N matrix (M sweeps, N samples each).
  3. Per-sweep dechirp by complex-multiplying the conjugate of the
     replica into each row.
  4. FFT in fast-time → range profile per sweep.
  5. FFT in slow-time → range-Doppler matrix.

The implementation is deliberately stdlib-numpy (no scipy) so the
client stays inside sigmond's "stdlib + ka9q-python + numpy" envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Speed of light, km/s.  Matches core/invert.py — keep them in sync.
C_KM_PER_S = 299_792.458


@dataclass(frozen=True)
class DechirpResult:
    """One coherent processing interval's dechirped range-Doppler output.

    Fields:
        range_doppler: complex M×N array — Doppler bins × range bins.
        range_axis_km: real N-vector — group range (km) per FFT bin.
        doppler_axis_hz: real M-vector — Doppler frequency (Hz) per slow-time bin.
        sweep_rate_hz_per_s: the sweep rate used (echoed for downstream consumers).
        sample_rate_hz: the sample rate used (likewise).
    """
    range_doppler: np.ndarray
    range_axis_km: np.ndarray
    doppler_axis_hz: np.ndarray
    sweep_rate_hz_per_s: float
    sample_rate_hz: float


def make_replica(
    n_samples: int,
    sample_rate_hz: float,
    sweep_rate_hz_per_s: float,
    *,
    window: bool = True,
    phase_offset_samples: int = 0,
) -> np.ndarray:
    """Generate one FMCW chirp replica at complex baseband.

    The transmitted CODAR chirp has linearly-varying instantaneous
    frequency ``f(t) = κ·t`` over one sweep period.  Phase = ∫f dt =
    ½·κ·t² so the complex baseband signal is::

        s(t) = exp(j · 2π · ½ · κ · t²)

    A Hann window is applied by default to suppress sidelobe artifacts
    when Fourier-transforming a finite-support function (Kaeppler Eq. 6).

    Args:
        n_samples: samples per sweep (= sample_rate_hz / sweep_repetition_hz).
        sample_rate_hz: IQ sample rate (Hz).
        sweep_rate_hz_per_s: signed sweep rate (Hz/s).  Sign matters: a
            CODAR down-chirp has ``κ < 0``; up-chirps have ``κ > 0``.
        window: apply a Hann window to suppress FFT sidelobes.
        phase_offset_samples: TDMA sweep-start phase, in samples within
            one sweep period.  When two CODAR transmitters share a band
            via TDMA, each starts its sweep at a different phase within
            the 1 s period; pass that phase here to build a replica that
            matches a specific transmitter.  The phase wraps modulo
            ``n_samples``: passing ``n_samples//4`` builds a replica that
            had been sweeping for a quarter-period when our window opens,
            so its instantaneous frequency at t=0 is ``κ · n_samples/4 /
            sample_rate_hz`` rather than 0.  Default 0 = standard
            zero-offset replica (compatible with v0.2 callers).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0; got {n_samples}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}")

    # Compute the time axis modulo one sweep period so the replica
    # wraps continuously: a TX whose sweep started ``phase_offset_samples``
    # samples ago is, at our t=0, ``phase_offset_samples / sample_rate_hz``
    # seconds into its sweep.  When that elapsed time crosses one full
    # period, the sweep restarts → the modulo handles the wrap cleanly.
    offset = int(phase_offset_samples) % n_samples
    t_into_sweep = ((np.arange(n_samples) + offset) % n_samples) / sample_rate_hz
    phase = 2.0 * np.pi * 0.5 * sweep_rate_hz_per_s * t_into_sweep * t_into_sweep
    replica = np.exp(1j * phase)
    if window:
        replica = replica * np.hanning(n_samples)
    return replica.astype(np.complex64)


def dechirp(
    rx_samples: np.ndarray,
    *,
    sample_rate_hz: float,
    sweep_rate_hz_per_s: float,
    sweep_repetition_hz: float,
    apply_window: bool = True,
    phase_offset_samples: int = 0,
) -> DechirpResult:
    """Dechirp a CPI of received IQ samples and return the range-Doppler matrix.

    Args:
        rx_samples: 1-D complex IQ from the radiod channel.  Must be at
            least M·N samples long (M = SRF·CPI seconds, N =
            sample_rate / SRF); excess samples at the tail are
            discarded.
        sample_rate_hz: IQ sample rate.  Must equal the rate of the
            radiod channel that produced ``rx_samples``.
        sweep_rate_hz_per_s: signed CODAR sweep rate.  This selects
            *which* CODAR transmitter is being dechirped — multiple
            transmitters in the same band (TDMA-offset) are
            distinguished by their sweep timing/rate.
        sweep_repetition_hz: number of sweeps per second.  Determines
            both the sweep period (and thus N) and the Doppler
            unambiguous bandwidth.
        apply_window: pass a Hann window through to the replica
            generator.
        phase_offset_samples: TDMA sweep-start phase for this TX (see
            ``make_replica``).  Selects which TX in a shared band to
            dechirp.  Default 0 = zero-offset replica (v0.2 behaviour).
    """
    if rx_samples.dtype.kind != "c":
        raise ValueError(
            f"rx_samples must be complex; got dtype={rx_samples.dtype}"
        )
    if sweep_repetition_hz <= 0:
        raise ValueError(
            f"sweep_repetition_hz must be > 0; got {sweep_repetition_hz}"
        )

    n_samples = int(round(sample_rate_hz / sweep_repetition_hz))
    if n_samples <= 1:
        raise ValueError(
            f"sample_rate_hz / sweep_repetition_hz = {n_samples} must be > 1"
        )
    n_sweeps = len(rx_samples) // n_samples
    if n_sweeps < 1:
        raise ValueError(
            f"need at least one full sweep ({n_samples} samples); "
            f"got {len(rx_samples)}"
        )

    # Reshape into M × N: rows are sweeps (slow time), columns are samples
    # within one sweep (fast time).
    rx_matrix = rx_samples[: n_sweeps * n_samples].reshape(n_sweeps, n_samples)

    # Build the replica and dechirp each sweep by multiplying with the
    # complex-conjugate.  The result has a beat tone for each propagation
    # mode at frequency = sweep_rate · path_delay.
    replica = make_replica(
        n_samples, sample_rate_hz, sweep_rate_hz_per_s,
        window=apply_window,
        phase_offset_samples=phase_offset_samples,
    )
    dechirped = rx_matrix * np.conj(replica)

    # FFT along the fast-time axis → range spectrum per sweep.
    range_spectrum = np.fft.fft(dechirped, axis=1)

    # FFT along the slow-time axis → Doppler resolution per range bin.
    range_doppler = np.fft.fftshift(np.fft.fft(range_spectrum, axis=0), axes=0)

    # Convert FFT bin → group range.  beat_freq = κ · τ → τ = beat_freq / κ.
    # For a CODAR down-chirp (κ < 0), positive delays produce negative
    # beat frequencies, so we use abs(κ) and take abs() of the bin
    # frequency to land on positive group range regardless of sweep
    # direction.
    bin_freqs_hz = np.fft.fftfreq(n_samples, d=1.0 / sample_rate_hz)
    delays_s = bin_freqs_hz / abs(sweep_rate_hz_per_s)
    range_axis_km = C_KM_PER_S * delays_s

    doppler_axis_hz = np.fft.fftshift(
        np.fft.fftfreq(n_sweeps, d=1.0 / sweep_repetition_hz)
    )

    return DechirpResult(
        range_doppler=range_doppler,
        range_axis_km=range_axis_km,
        doppler_axis_hz=doppler_axis_hz,
        sweep_rate_hz_per_s=sweep_rate_hz_per_s,
        sample_rate_hz=sample_rate_hz,
    )


def range_profile(result: DechirpResult) -> np.ndarray:
    """Sum |range_doppler|² across Doppler bins → 1-D power-vs-range vector.

    For F-region detection we don't need Doppler resolution — the dominant
    energy at each range integrates across all Doppler bins.  Returns
    real-valued power in arbitrary units (consumer normalises if needed).
    """
    return np.abs(result.range_doppler).sum(axis=0).astype(np.float32)


def positive_range_window(
    result: DechirpResult, profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only the positive-range half of the range axis + profile.

    The FFT output covers ±half the unambiguous range; for sky-wave
    propagation we care only about positive group ranges (the path
    delay is always positive).  This crops the negative half cleanly.
    """
    pos_mask = result.range_axis_km >= 0
    # Sort by ascending range so plotting/peak-finding is well-behaved.
    order = np.argsort(result.range_axis_km[pos_mask])
    return (
        result.range_axis_km[pos_mask][order],
        profile[pos_mask][order],
    )

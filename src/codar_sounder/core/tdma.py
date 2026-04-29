"""TDMA offset discovery — separate co-band CODAR transmitters.

Multiple CODAR transmitters frequently share a band (e.g. DUCK + HATY
at 4.537 MHz, LISL + ASSA + CEDR at 4.575 MHz).  In SeaSonde TDMA
deployments, each TX is GPS-disciplined to start its 1 s sweep at a
distinct phase offset within the second; in that case a phase-aligned
replica per TX cleanly extracts that TX's returns and suppresses the
others.

This module discovers the sweep-start phase offsets present in a
captured IQ buffer by cross-correlating it with a zero-offset chirp
replica.  Each TDMA-distinct TX produces a sharp cross-correlation peak
at its direct-path arrival time within the period.

The output is a list of ``(offset_samples, snr_db)`` tuples that the
daemon can:

  - log for an operator to inspect (``codar-sounder tdma-scan``);
  - convert to per-TX ``tdma_offset_samples`` config entries by
    subtracting each TX's known direct-path delay (= ground distance /
    speed of light) and rounding to the nearest sample;
  - feed back to ``dechirp(..., phase_offset_samples=...)`` per TX so
    each TX's returns are extracted with cross-suppression of the
    others.

If the TXs in a band are not TDMA-distinct (e.g. simultaneous-broadcast
FDMA at sub-kHz spacing), discovery returns one dominant peak; phase-
offset dechirping cannot separate them in that case and a different
discrimination technique is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dechirp import C_KM_PER_S, make_replica


@dataclass(frozen=True)
class TDMAPeak:
    """One cross-correlation peak from TDMA discovery."""
    offset_samples: int           # sweep-start phase, samples within one period
    snr_db: float                  # peak / median(corr power), in dB
    correlation_power: float       # raw |corr|² at the peak (arbitrary units)


def discover_tx_offsets(
    rx_samples: np.ndarray,
    *,
    sample_rate_hz: float,
    sweep_rate_hz_per_s: float,
    sweep_repetition_hz: float,
    snr_threshold_db: float = 10.0,
    min_separation_samples: int = 32,
    max_peaks: int = 8,
) -> list[TDMAPeak]:
    """Return cross-correlation peaks of one rx period against a zero-offset replica.

    Each peak's ``offset_samples`` is the time-of-arrival of one TX's
    direct path within the period.  To convert into a TDMA sweep-start
    phase suitable for ``dechirp(phase_offset_samples=...)``, the daemon
    subtracts the TX's known direct-path delay (ground distance / c) —
    see ``offset_for_tx`` below.

    Args:
        rx_samples: 1-D complex IQ.  Must be at least one sweep period
            long; if longer, is averaged across periods to suppress
            sky-wave fading.
        sample_rate_hz: IQ sample rate (Hz).  Must match the radiod
            channel that produced ``rx_samples``.
        sweep_rate_hz_per_s: sweep rate of the band — same for every TX
            in a shared band.
        sweep_repetition_hz: SRF (sweeps per second).  All TXs in a band
            share this, so the search space is ``int(sample_rate / SRF)``
            samples = one sweep period.
        snr_threshold_db: minimum (peak / median) ratio for a peak to
            count.  10 dB is conservative — direct-path peaks in a clean
            band typically run 20–40 dB above the noise floor.
        min_separation_samples: peaks closer than this are merged (the
            stronger one wins).  At 64 kHz / SRF=1 Hz, 32 samples ≈ 0.5 ms
            ≈ 150 km — well below typical TDMA slot spacing but above
            the cross-correlation main-lobe width.
        max_peaks: cap on number of peaks returned, sorted by SNR
            descending.

    Returns: list of ``TDMAPeak``, sorted by SNR descending.
    """
    if rx_samples.dtype.kind != "c":
        raise ValueError(f"rx_samples must be complex; got {rx_samples.dtype}")

    n_per_sweep = int(round(sample_rate_hz / sweep_repetition_hz))
    if len(rx_samples) < 2 * n_per_sweep:
        raise ValueError(
            f"rx_samples length {len(rx_samples)} < two sweep periods "
            f"({2 * n_per_sweep} samples) — discovery needs multiple "
            f"periods for SNR averaging"
        )

    # Linear cross-correlation across multiple sweep periods.  A chirp
    # captured continuously wraps somewhere in our buffer; *single-period*
    # circular correlation handles this poorly because a one-period buffer
    # contains [tail_of_one_sweep, head_of_next] — a non-contiguous chirp
    # whose autocorrelation is washed out.  *Multi-period linear*
    # correlation against a one-period replica always finds at least one
    # full unbroken sweep within the buffer, regardless of where the wrap
    # falls; peaks recur every ``n_per_sweep`` lags, so we fold across
    # periods to coherently boost SNR.
    n_periods = len(rx_samples) // n_per_sweep
    rx_use = rx_samples[: n_periods * n_per_sweep]

    # Zero-offset, no-window replica — matched filter for the direct path.
    # Using window=False here keeps the cross-correlation main lobe narrow
    # (timing precision matters more than sidelobe suppression at this
    # stage; the daemon's dechirp() applies a window for the actual FFT).
    replica = make_replica(
        n_per_sweep, sample_rate_hz, sweep_rate_hz_per_s,
        window=False, phase_offset_samples=0,
    )

    # FFT-based linear correlation: zero-pad rep to len(rx_use) and
    # multiply spectra.  ``ifft(F_rx · conj(F_rep))`` gives the circular
    # cross-correlation; the first ``len(rx_use) - n_per_sweep + 1`` lags
    # are the *linear* cross-correlation values.
    rep_padded = np.zeros(len(rx_use), dtype=np.complex64)
    rep_padded[:n_per_sweep] = replica

    corr_full = np.fft.ifft(
        np.fft.fft(rx_use) * np.conj(np.fft.fft(rep_padded))
    )
    n_valid = len(rx_use) - n_per_sweep + 1
    corr_valid = corr_full[:n_valid]
    power_valid = np.abs(corr_valid) ** 2

    # The TX is periodic, so each TX produces a peak at every period
    # within the lag axis.  Fold to boost SNR: reshape into
    # (n_full_periods × n_per_sweep) — discarding the trailing partial
    # period — and accumulate power across the leading axis.  Result:
    # one accumulated row of length n_per_sweep, indexed by offset within
    # one period.
    n_full = n_valid // n_per_sweep
    if n_full < 1:
        # Should not happen given the >= 2 * n_per_sweep guard above.
        return []
    folded = power_valid[: n_full * n_per_sweep].reshape(n_full, n_per_sweep)
    accum_power = folded.sum(axis=0).astype(np.float64)

    median = float(np.median(accum_power))
    if median <= 0:
        return []

    # Local-maximum scan with min-separation enforcement.  We sort by
    # power descending and greedily admit peaks that aren't within
    # ``min_separation_samples`` of an already-admitted peak — modulo
    # n_per_sweep so peaks near the wrap point are handled correctly.
    candidate_idx = np.argsort(accum_power)[::-1]
    accepted: list[int] = []
    for idx in candidate_idx:
        idx = int(idx)
        snr_db = 10.0 * np.log10(accum_power[idx] / median)
        if snr_db < snr_threshold_db:
            break              # remaining candidates are weaker
        too_close = False
        for a in accepted:
            d = abs(idx - a)
            d = min(d, n_per_sweep - d)            # circular distance
            if d < min_separation_samples:
                too_close = True
                break
        if not too_close:
            accepted.append(idx)
        if len(accepted) >= max_peaks:
            break

    return [
        TDMAPeak(
            offset_samples=idx,
            snr_db=float(10.0 * np.log10(accum_power[idx] / median)),
            correlation_power=float(accum_power[idx]),
        )
        for idx in accepted
    ]


def offset_for_tx(
    correlation_peak_samples: int,
    *,
    ground_distance_km: float,
    sample_rate_hz: float,
    n_per_sweep: int,
) -> int:
    """Convert a discovered cross-correlation peak into a TDMA phase offset.

    The cross-correlation peak position is the time-of-arrival of the
    TX's direct path within one period.  The TX's sweep *started* a bit
    earlier — by the direct-path propagation delay (D / c).  Subtracting
    that delay yields the TDMA sweep-start phase, which is the value
    ``dechirp(phase_offset_samples=...)`` expects.

    Args:
        correlation_peak_samples: peak index from ``discover_tx_offsets``.
        ground_distance_km: TX → RX great-circle distance (km).
        sample_rate_hz: IQ sample rate.
        n_per_sweep: samples per sweep period (= sample_rate / SRF).
    """
    direct_delay_samples = int(round(
        ground_distance_km / C_KM_PER_S * sample_rate_hz
    ))
    return (correlation_peak_samples - direct_delay_samples) % n_per_sweep


def match_peaks_to_txs(
    peaks: list[TDMAPeak],
    tx_distances_km: dict[str, float],
    *,
    sample_rate_hz: float,
    n_per_sweep: int,
) -> dict[str, int | None]:
    """Greedy nearest-peak assignment of TXs to discovered offsets.

    For each TX, computes the expected correlation-peak position from
    its known ground distance and an unknown TDMA offset, then assigns
    the closest unassigned discovered peak.  Returns a dict of
    ``tx_id -> tdma_offset_samples`` (suitable for
    ``dechirp(phase_offset_samples=...)``), with ``None`` for any TX
    that couldn't be matched (no peak within ``min_separation_samples``
    of plausible).

    Note: this is a heuristic.  When the number of peaks exceeds the
    number of TXs (e.g. spurious clutter peaks), the lowest-SNR
    candidates may be ignored.  When the number of peaks is fewer, some
    TXs return ``None`` — the daemon should fall back to a zero-offset
    dechirp for those and flag ``tdma_locked: false`` in the JSONL.

    For v0.3 the matching is intentionally simple: assignment by closest
    peak in time-of-arrival, no joint optimization.  Co-incident TXs
    (TDMA peaks within ``min_separation_samples`` of each other) cannot
    be reliably separated by this scheme — that's a known limitation.
    """
    # All TXs in a band share the same TDMA period, but their direct-
    # path delays differ by their geometric ground distances.  We expect
    # the ranking of correlation peaks (by offset_samples mod period) to
    # match the ranking of TXs by some-permutation-of (ground delay +
    # TDMA slot).  Without prior knowledge of slot ordering, the cleanest
    # heuristic is: sort TXs by ground distance, sort peaks by
    # correlation_power desc (proxy for "closer / brighter TX"), pair in
    # order.  For the canonical 4.575 MHz ODU cluster (LISL ~closest,
    # ASSA, CEDR) at our St-Louis receiver this gets the assignment
    # right when all three TXs are detected.
    sorted_txs = sorted(tx_distances_km.items(), key=lambda kv: kv[1])
    sorted_peaks = sorted(peaks, key=lambda p: -p.snr_db)

    out: dict[str, int | None] = {tx: None for tx in tx_distances_km}
    for (tx_id, D), peak in zip(sorted_txs, sorted_peaks):
        out[tx_id] = offset_for_tx(
            peak.offset_samples,
            ground_distance_km=D,
            sample_rate_hz=sample_rate_hz,
            n_per_sweep=n_per_sweep,
        )
    return out

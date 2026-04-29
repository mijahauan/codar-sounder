# v0.3 — TDMA-aware dechirping

## Problem

Multiple CODAR transmitters share each band via TDMA-style sweep-start time
offsets:

- 4.537 MHz: DUCK, HATY (UNC group)
- 4.575 MHz: LISL, ASSA, CEDR (ODU group)
- 4.513 MHz: BLCK, BRIG, HOOK, LOVE, MRCH, MVCO, NANT, WILD (Rutgers group)
- 4.785 MHz: BMLR, SHEL, MAN1, PSG1, WIN1, YHL1 (West Coast)
- ... etc

All TXs in a band use the same sweep rate κ and SRF (1 Hz typical), so a
single replica dechirp cannot distinguish them.  Today v0.2 dechirps once
per CPI and reports the same group ranges for every TX in the config —
actively wrong output when a band has multiple TXs.

## Approach

Per-TX phase-offset replica.  Each TX has a unique sweep-start time offset
within the 1 s sweep period.  Build a replica that wraps the chirp at that
TX's offset; dechirp the same IQ once per TX; attribute peaks to the TX
whose replica produced them.

Offsets are GPS-disciplined and stable, so we discover-once-cache-forever
(re-lock daily or on signal loss).  Discovery: cross-correlate the IQ with
a single zero-offset replica — peaks in the cross-correlation give each
TX's sweep-start time within the period.

## Tasks

- [x] Write plan
- [x] `core/dechirp.py` — extend `make_replica()` with `phase_offset_samples`
      and `dechirp()` to pass it through.  Replica wraps modulo sweep period.
- [x] `core/tdma.py` (new) — `discover_tx_offsets(rx_samples, ...)` returns
      list of (offset_samples, snr_db) for each TX in the band.  Match
      offsets to known TXs by ascending ground-distance order.
- [x] `core/daemon.py` — read optional `tdma_offset_samples` per
      `[[radiod.transmitter]]` (default 0 = v0.2 behaviour).  Pass through
      to `dechirp(phase_offset_samples=...)`.
- [x] `cli.py` — new `codar-sounder tdma-scan` subcommand: capture IQ for
      N seconds, run discovery, print per-TX assignments.  Operator
      pastes the values into config.
- [x] `tests/test_dechirp.py` — extend `_synth_chirp` with
      `target_tdma_offsets_s`.  Add tests that two TXs at different offsets
      produce peaks in the *correct* replica's output and ≥10 dB
      cross-suppression in the *wrong* replica's output.
- [x] `tests/test_tdma.py` (new) — synthetic two-TX TDMA IQ →
      `discover_tx_offsets()` returns both offsets within ±2 samples.
- [x] Run full suite — 87 passed, 24 skipped (Kaeppler dataset; pre-existing).
- [x] Smoke-test `tdma-scan` on bee1-rx888 with the 4.575 MHz config —
      machinery runs end-to-end.  **Live discovery returned only 5 dB-SNR
      peaks at this hour, well below useful threshold.**  Either the ODU
      group isn't TDMA-distinguishable via single-period cross-correlation,
      or the band is too quiet right now.  See "Field results" below.
- [ ] Bump pyproject + deploy.toml to 0.3.0; commit; push.

## Field results — 2026-04-29 evening

`codar-sounder tdma-scan --radiod-id ac0g-bee1-rx888 --seconds 10` ran
to completion against the 4.575 MHz channel.  Top peak: 5 dB above
median (correlation_power ~3e-4).  Real TDMA peaks would be 20–40 dB.
Two interpretations, neither yet refuted:

1. **The ODU LISL/ASSA/CEDR group transmits simultaneously** (FDMA at
   sub-kHz spacing within the 25 kHz BW, or co-incident sweep starts).
   Phase-offset dechirping cannot separate them in either case.  A
   cross-loop antenna (deferred to v0.4) gives AOA discrimination,
   which would separate them by bearing.

2. **The single-period cross-correlation is fragile to chirp wrap
   boundaries.**  A multi-period linear correlation (using rx of
   length 2T against a replica of length T) might lock more reliably.
   This is a v0.4 refinement.

Either way, **v0.3 doesn't auto-promote discovered offsets to the
daemon's runtime path** — the operator runs `tdma-scan`, reads the
output, and decides whether to add `tdma_offset_samples` to each
`[[radiod.transmitter]]` block.  The daemon honours those values when
present (falling back to 0 = v0.2 behaviour when absent).

## Out of scope (explicitly deferred)

- Cross-loop O/X polarization (Stokes V) — needs second physical antenna.
  Owner has agreed to procure one; revisit as v0.4.
- Re-lock cadence beyond "once per 24 h or on signal loss".  No predictive
  drift model — TXs are GPS-locked, drift is negligible.
- HFRNet TDMA-table import.  Self-discovery is sufficient and avoids an
  external-data dependency.

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
- [x] Bump pyproject + deploy.toml to 0.3.0; commit; push.

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


## v0.4.0 — multi-peak + layer classification + CH sink (2026-05-07)

Closes the v0.3 "deferred to v0.4" list except the cross-loop AOA item,
which stays out until a second antenna is procured.

### Tasks

- [x] `core/stream.py` — wire ka9q-python ≥3.11 `low_edge`/`high_edge`
      kwargs into `RadiodIQSource.ensure_channel`.  Default ±sample_rate/2
      ∓ 1500 Hz guard.  Optional override via constructor kwargs (kept
      out of config until field SNRs prove a per-station tightness is
      worth the knob).  Bumped ka9q-python pin to >=3.11.0.
- [x] `core/trace.py` — `find_f_region_peaks` (plural).  Local-max
      scan with SNR threshold + minimum-separation collapse; sorted
      by SNR descending; capped at `max_peaks` (default 4).  The old
      singular `find_f_region_peak` becomes a thin wrapper for
      backwards compat.
- [x] `core/invert.py` — `classify_layer(virtual_height_km)` returns
      one of `E`/`F1`/`F2`/`F2_extreme`/`below_E`/`unknown` per Davies
      (1990) digisonde altitudes.  `IonosphericFix` gains a
      `mode_layer` field set automatically by `invert()`.
- [x] `core/output.py` JSONL — adds `peak_index`, `peak_count`,
      `mode_layer` fields per record.
- [x] `core/daemon.py` `process_cpi` — emits one record per peak (was:
      single argmax peak); shared per-radiod CH writer initialised
      from `sigmond.hamsci_ch.Writer.from_env`; per-peak CH inserts
      run alongside the JSONL writes.  CH path failure is non-fatal.
- [x] `clickhouse/schema/codar/{000,001}_*.sql` — greenfield `codar`
      database; `codar.spots` is ReplacingMergeTree, monthly-partitioned,
      ORDER BY `(host_call, station_id, time, peak_index)`.
- [x] `deploy.toml` — bumped contract_version to `0.6`, version to
      `0.4.0`, added `[clickhouse]` block referencing
      `clickhouse/schema/codar`.
- [x] `contract.py` — bumped CONTRACT_VERSION to `0.6`; replaced
      `disk_writes` with `data_sinks` (file always; clickhouse appears
      when `SIGMOND_CLICKHOUSE_URL` is set).
- [x] `cli.py` `tdma-scan --write-config` — atomic in-place TOML edit
      that persists discovered offsets, replacing existing
      `tdma_offset_samples` lines or inserting new ones after the
      matching `id = "..."`.  Comments and unrelated formatting
      preserved.
- [x] Tests:
      - `test_stream.py` — 6 tests on filter-edge defaulting.
      - `test_multi_peak.py` — 25 tests covering `classify_layer`,
        `find_f_region_peaks`, per-peak CH row builder, end-to-end
        synthetic CPI emission to a fake CH writer.
      - `test_tdma_config_writer.py` — 9 tests on the in-place
        TOML rewriter (replace + insert + atomic-write + scope).

### Out of scope (still / again)

- **Cross-loop / crossed-dipole AOA** — needs a second physical antenna.
  Owner has only one; revisit when a second antenna arrives.
- **Dynamic TDMA re-lock** — confirmed unnecessary: CODAR TXs are
  GPS-disciplined, drift is negligible at the timescales we care about.
- **Auto-promoted TDMA offsets at daemon startup** — the field-test
  retro called this risky without operator review.  v0.4 adds
  `--write-config` so the operator can promote in one keystroke after
  eyeballing SNRs (typical real TDMA peaks are 20–40 dB; 5 dB is noise).
- **HFRNet table import** — self-discovery is sufficient; an
  external-data dependency would be a regression.

### Verification status

Unit tests: 158 passed locally (was 117 in v0.3.2; +41 new).  Live
verification on bee1-rx888 with the wideband filter: TBD post-deploy.
Expected: SNRs that were 5 dB on the 4.5 MHz band in v0.3 should rise
to 20+ dB now that the chirp is no longer being truncated by the
default ±5 kHz audio filter.


## v0.5.0 — ITU-R P.531 scintillation indices (2026-05-20)

Closes the identified gap recorded as
`project_codar_sounder_scintillation_gap` in memory: codar-sounder
already produces, but discards, the per-CPI per-mode complex amplitude
time series.  Adding S4 + σ_φ extends the L1 product to a
propagation-mode-resolved scintillation index — a companion to
hf-timestd's WWV-tone-only scintillation, with oblique geometry and
multiple modes per CPI.

### Tasks

- [x] Write plan (`/home/mjh/.claude/plans/functional-floating-garden.md`).
- [x] `core/scintillation.py` (new) — `ScintillationResult` dataclass
      (8 fields) + `compute_scintillation()`.  ITU-R P.531 severity
      bins (strict-less-than); event gate at S4 ≥ 0.3 or σ_φ ≥ 0.2;
      `confidence = min(1, n_samples/30)` with NaN/Inf guard; zero-
      signal short-circuit at `mean_intensity < 1e-30`.
- [x] `core/dechirp.py` — `DechirpResult` gains
      `range_spectrum: np.ndarray` (complex64, M×N, the pre-Doppler-
      FFT matched-filter output).  Cast to complex64 once after the
      fast-time FFT so the per-CPI memory cost is halved vs. numpy's
      default complex128.  Add `positive_to_raw_index_map(result)` and
      `raw_bin_from_positive(result, idx)` for the
      positive-sorted → raw FFT-bin lookup the scintillation slice
      needs.
- [x] `core/output.py` `JsonlWriter.write` — required new
      `scintillation` kwarg.  Adds 8 fields to the JSONL record.
      `s4_index` and `sigma_phi_rad` written full-precision (no
      rounding) so downstream consumers can reproduce the severity
      bin deterministically.  Schema docstring bumped to v0.5.
- [x] `core/daemon.py` `process_cpi` — compute the positive→raw
      index map once per CPI; per peak slice
      `range_spectrum[:, raw_indices[detection.bin_index]]` and pass
      the resulting `ScintillationResult` through to `JsonlWriter.write`
      and `_ch_row_for`.  Per-peak log line gains S4 + σ_φ + EVENT
      marker.
- [x] `core/daemon.py` `_ch_row_for` — gains `scintillation` kwarg;
      8 new fields with explicit casts (no rounding, matching the
      existing convention).
- [x] `tests/test_scintillation.py` (new) — 55 tests covering
      pure-CW baseline, S4 closed-form recovery, σ_φ closed-form
      recovery (period-4 phase pattern orthogonal to the linear
      detrend by construction), severity-helper unit tests at exact
      float64 boundaries, severity end-to-end tests offset 1e-4 from
      the boundary, event-gate triggers, quality gating (min_samples,
      zero signal, NaN/Inf), sample-rate scaling, dataclass field
      lock.
- [x] `tests/test_dechirp.py` — `range_spectrum` shape + complex64
      dtype assertions; `raw_bin_from_positive` round-trip; slow-time
      column at peak bin is finite complex64 M-vector.
- [x] `tests/test_multi_peak.py` — `expected_cols` schema set updated
      with the 8 new fields; row round-trip values asserted; daemon-
      level integration test asserts the scintillation fields land on
      the synthetic CPI's sink row with sane values
      (`scintillation_event` is False, severity classified, samples
      ≥ 10).
- [x] `README.md` — v0.5.0 highlights section (8 new fields, severity
      bins, cadence caveat, confidence model, no contract bump).
- [x] `pyproject.toml` + `deploy.toml` — version `0.4.0` → `0.5.0`;
      `contract_version` stays `0.6`.
- [x] Live verification on bee1-rx888 — superseded by the
      v0.5.1 + v0.5.2 cycle below; live deployment exposed real
      issues (single-bad-sweep contamination, ITU-R-vs-HF threshold
      mismatch) that drove two patches.  End-to-end data path
      verified; multi-day Kp correlation remains an open analysis
      task (not a coding task).

### Out of scope (deferred)

- **Cross-CPI rolling-window indices** — per-CPI is the v0.5 scope.
  Multi-CPI integration adds peak-bin tracking jitter (the peak
  migrates ±1 bin between CPIs); revisit if field data shows per-CPI
  noise dominates real scintillation signal.
- **0.1 Hz canonical ITU-R high-pass** — would need slow-time
  oversampling beyond the 1 Hz SRF, i.e. a fundamental architecture
  change.  The 1/CPI ≈ 0.017 Hz effective corner is documented in
  README so consumers don't cross-compare to GNSS σ_φ blind.
- **3-bin power-sum smoothing** — would bias S4 toward "weak" by
  integrating off-target noise.  Wrong for per-CPI; the right answer
  is per-peak single-bin within one CPI's matched-filter output.

### Verification status

Unit tests: 193 passed locally (was 158 in v0.4.0; +59 new — 56 in
`test_scintillation.py`, 3 in `test_dechirp.py` extensions, plus
schema-set extension in `test_multi_peak.py`'s
`test_row_columns_match_codar_spots_schema`).  24 pre-existing
Kaeppler Zenodo-dataset skips unchanged.


## v0.5.1 — MAD outlier rejection (2026-05-21)

Live verification on bee1-rx888 SEAB (13.45 MHz, CPI=15s) immediately
revealed v0.5.0 was producing **100% strong-event rate** on every
peak.  Root cause via a live-IQ probe: one sweep per CPI carried
broadband spectral leakage (FFT peak in the negative-range half — an
unusable matched-filter row, likely from an RFI burst or ka9q packet
duplication).  That bad sweep contributed one anomalously-large
intensity sample into every range bin's slow-time vector, inflating
S4 to ≈ √(M-1) ≈ 3.7 at M=15.

### Tasks

- [x] `core/scintillation.py` — add MAD-based outlier rejection in
      ``compute_scintillation`` before computing S4/σ_φ.  Reject
      samples with ``|I - median(I)| > 4·MAD(I)``; fall back to
      ``1.2533·MeanAD`` when MAD = 0 (Iglewicz-Hoaglin 1993).  Add
      ``n_outliers_rejected`` to ``ScintillationResult``; report
      retained count in ``n_samples``.  Re-check the ``min_samples``
      floor against the retained count (returns "unknown" if
      rejection drops below).
- [x] `core/output.py` + `core/daemon.py` — surface
      ``scintillation_outliers_rejected`` in JSONL records,
      hamsci_sink rows, and the per-peak log line (``n=14-1`` style
      marker).
- [x] Tests — 6 new tests covering single-outlier rejection, multiple
      outliers, no-outliers-on-clean, MAD=0 fallback, rejection
      below floor → unknown, field-simulation reproducing the
      production bad-sweep pattern.
- [x] Update `tests/test_multi_peak.py` `expected_cols` set.
- [x] `pyproject.toml` + `deploy.toml` — version 0.5.0 → 0.5.1.

### Verification status

199 tests pass (was 193 in v0.5.0).  Live verification: MAD
rejection fires 0-5 times per peak per CPI (mode 2 — matching the
"two adjacent RFI burst sweeps" the probe found later).  S4 still
mostly strong but moved from "always 3+" to "0.5-1.1 range"
(physically plausible now).


## v0.5.2 — quadratic detrend + HF-recalibrated σ_φ thresholds (2026-05-21)

After v0.5.1 fixed S4, σ_φ was still flagging strong everywhere.
A second probe captured 4 real F-region peaks at 60 dB SNR and
showed:

  - No Doppler aliasing (0 of 64 phase steps > π).
  - Linear detrend underfits for peaks with curved phase
    trajectories — quadratic detrend reduces σ_φ by 25-60% on those.
  - Even with perfect detrending, HF oblique multipath produces an
    intrinsic σ_φ floor of ~0.4-0.6 rad on quiet days — ITU-R
    P.531's 0.2/0.5 thresholds (calibrated for single-mode GNSS/SHF)
    misclassify it as "moderate"/"strong".

### Tasks

- [x] `core/scintillation.py`:
      - Change `polyfit(times, phases, deg=1)` → `deg=2`, with
        ``times`` centered before fitting so the linear coefficient
        is the average-Doppler slope at the CPI centroid.
      - Update ``mode_doppler_hz`` extraction: ``coeffs[1] / (2π)``
        (was ``coeffs[0]`` for deg=1).
      - Move ``SIGMA_PHI_WEAK_MAX``: 0.2 → 0.5; ``SIGMA_PHI_MODERATE_MAX``:
        0.5 → 1.0; ``SIGMA_PHI_EVENT_THRESHOLD``: 0.2 → 0.5.
      - Update module docstring with the HF-deviation rationale and
        the live-data evidence table.
- [x] `tests/test_scintillation.py`:
      - Replace ``_phase_pattern_orthogonal_to_linear`` with the new
        period-4 pattern ``(1/√5)·[-1, +3, -3, +1]`` (orthogonal to
        constant, linear, *and* quadratic over each 4-block).
      - Update boundary test parametrizations for the new thresholds.
      - Loosen the doppler-trend test tolerances (2e-2 rad for σ_φ,
        1e-4 Hz for doppler) to reflect complex64 precision at the
        38-rad phase range.
- [x] `README.md` — v0.5.2 highlights with the HF-recalibrated
      thresholds + rationale.
- [x] `pyproject.toml` + `deploy.toml` — version 0.5.1 → 0.5.2.

### Verification status

199 tests pass.  Live verification post-deploy:

  - **Mixed quiet F2 (h' ~ 500 km)**: σ_φ_quadratic ≈ 0.45-0.95 →
    weak / moderate (3 of 4 probe peaks).
  - **Disturbed F2_extreme (h' > 600 km)**: σ_φ_quadratic ≈ 1.2-1.8
    → strong.
  - Production daemon sees mostly F2_extreme right now (high local
    event rate consistent with real disturbed ionospheric conditions
    rather than calibration error; cross-check Kp/SWPC to confirm).
  - MAD rejection continues to fire 1-3× per peak.
  - All 9 scintillation fields present in JSONL + sink rows.


## v0.6.0 — σ_φ diagnostic fields (2026-05-21)

Follows up the v0.5.2 finding that linear detrend underfits real
F-region peaks with curved slow-time phase trajectories (TIDs,
multipath beating, accelerating Doppler).  Rather than choosing
linear *or* quadratic and hiding the other, expose both as wire
fields with the ratio as a self-contained underfit detector — a
TID/multipath-beating signature independent of the σ_φ severity
classification.

### Tasks

- [x] `core/scintillation.py`:
      - Compute both linear-detrend and quadratic-detrend σ_φ on each
        slow-time vector.
      - Canonical ``sigma_phi_rad`` stays = quadratic (matches v0.5.2
        production behaviour exactly — no break for downstream
        readers).
      - ``ScintillationResult`` gains ``sigma_phi_linear_rad``,
        ``sigma_phi_quadratic_rad``, ``sigma_phi_underfit_ratio``
        (= linear / quadratic; ≥ 1 by construction).
      - Pathological branches (degenerate fit, unknown result) return
        ratio = 1.0 by convention.
- [x] `core/output.py` + `core/daemon.py` — 3 new wire fields on
      JSONL records and ``codar.spots`` rows.
- [x] Tests:
      - 6 new ``TestUnderfitRatio`` tests covering: unity for pure
        CW; unity for constant Doppler; >> 1 for purely-quadratic
        phase; canonical = quadratic; ratio ≥ 1 across random
        inputs; unknown-result fallback.
      - ``test_multi_peak.py`` schema-set extension + integration
        assertions.
- [x] `README.md` v0.6.0 highlights.
- [x] `pyproject.toml` + `deploy.toml` — version 0.5.2 → 0.6.0.
      `contract_version` stays 0.6 (additive payload-schema only).

### Out of scope (still / again)

- **Multi-day Kp / SWPC baseline analysis** — confirm v0.5.2 σ_φ
  thresholds match real ionospheric activity over a week+ of
  observations.  Mostly an analysis task (not a code change).
- **Cross-CPI rolling-window scintillation** — smoother indices,
  adds peak-bin-tracking state.  Reconsider after underfit_ratio
  field-data gives us a sense of typical curvature scales.
- **Per-CPI RFI burst root cause** — currently masked by v0.5.1 MAD;
  upstream cause unknown (RFI?  ka9q packet duplication?  RX888
  saturation?).

### Verification status

205 unit tests pass (was 199 in v0.5.2; +6 new).  24 pre-existing
Kaeppler Zenodo-dataset skips unchanged.  Live verification on
bee1-rx888: TBD post-deploy — expect ``sigma_phi_underfit_ratio``
values to cluster around 1.0-1.5 on quiet F2 paths and >> 2 on
disturbed F2_extreme paths (per the probe data from 2026-05-21
showing linear → quadratic reductions of 25-60%).


## v0.6.1 — per-sweep MAD pre-filter (2026-05-21)

Investigation of the per-CPI bad-sweep artifact (deferred from
v0.5.1) found that ~80% of CPIs at 13.45 MHz on bee1-rx888 had 1-2
contaminated sweeps split across three distinct populations:

  - Broadband impulses (~60% of bursts): spread ~17-18 kHz of the
    passband; 3-7 ms duration; sferic-like (lightning, ESD,
    switching transients).
  - Discrete-tone RFI (~25% of bursts): peak/median 30-66, peak
    frequency ≈ -20 kHz offset from center (= 13.430 MHz absolute,
    in the 22-meter shortwave broadcast band).
  - Persistent multi-row disturbances (~15%): >1 second duration,
    contaminating multiple consecutive sweeps.

v0.5.1's per-peak MAD on slow-time intensity catches per-bin
amplitude outliers but doesn't address sweep-level contamination
that touches every range bin.  A per-sweep MAD test on post-fast-
time-FFT spectrum-total power catches all three populations.

### Tasks

- [x] `core/dechirp.py`:
      - Add ``SWEEP_MAD_REJECTION_K = 4.0`` constant.
      - After computing the complex64 ``range_spectrum``, compute per-
        sweep total power, MAD-test against the CPI median (with
        Iglewicz-Hoaglin MeanAD fallback for the MAD=0 degenerate
        synthetic-test case), and zero out outlier sweeps in-place
        before deriving ``range_doppler``.
      - ``DechirpResult`` gains ``n_sweeps_rejected: int`` field
        (default 0 for backward-compat with any callers that
        construct it directly).
- [x] `core/output.py` + `core/daemon.py`:
      - ``JsonlWriter.write`` and ``_ch_row_for`` thread a new
        ``dechirp_sweeps_rejected`` kwarg through to the wire fields.
      - Daemon emits a single ``log.info`` per CPI when the pre-filter
        fires (suppressed when 0).
- [x] Two-stage MAD coordination (folded into v0.6.1 after live
      verification revealed misses where embedded zeros in the slow-
      time vector polluted the per-peak MAD scale, masking their
      own outlier status):
      - ``DechirpResult`` gains ``bad_sweep_mask: Optional[np.ndarray]``
        (1D bool, length M).
      - ``compute_scintillation`` gains ``pre_rejected_mask`` kwarg:
        used as the initial keep mask before the per-peak MAD;
        statistics are computed only on the upstream-kept samples so
        the zeros can't drag the median or inflate MAD.
      - ``process_cpi`` passes ``result.bad_sweep_mask`` through.
      - ``n_outliers_rejected`` counts only the per-peak MAD's *extra*
        rejections, not the upstream pre-rejected count — the two
        stages report their work separately.
- [x] Tests:
      - 6 ``TestPerSweepMADRejection`` cases in test_dechirp.py:
        clean CPI → 0 rejected; one injected bad sweep → exactly 1
        rejected (and the row is zeroed); multiple injected bad
        sweeps → exact count rejected; bad sweep does NOT corrupt
        range_profile peak detection; ``n_sweeps_rejected`` field
        present on DechirpResult; ``bad_sweep_mask`` matches
        ``n_sweeps_rejected`` and aligns with zeroed rows.
      - 3 ``TestMADOutlierRejection`` cases for the ``pre_rejected_mask``
        kwarg: upstream zeros excluded from per-peak MAD math;
        additional per-peak outliers still detected; dimension
        mismatch raises ``ValueError``.
      - ``test_multi_peak.py``: ``expected_cols`` extended with
        ``dechirp_sweeps_rejected``; ``_ch_row_for`` test passes the
        new kwarg; integration test asserts the field is int and ≥ 0.
- [x] `README.md` v0.6.1 highlights.
- [x] `pyproject.toml` + `deploy.toml` — version 0.6.0 → 0.6.1.

### Verification status

214 tests pass (was 205 in v0.6.0; +9 new total — 5 dechirp MAD
filter + 3 scintillation coordination + 1 mask-shape verification).
24 pre-existing Kaeppler Zenodo-dataset skips unchanged.

First-deploy verification (between coordination fix and final
deploy) showed live data with the bad_sweep_mask propagation
working correctly: when ``dechirp_sweeps_rejected = N``, the
per-peak slow-time vectors at those positions are excluded from
the per-peak MAD statistics so a wider-variance bin's intrinsic
fluctuation can't mask the upstream rejection.


## Kp-baseline analysis (2026-05-21)

Analysis-only task (no codar-sounder code change) — added a
re-runnable correlation script and saved the first report:

  - ``scripts/kp_correlation_analysis.py`` — pulls NOAA SWPC
    planetary-Kp JSON, glob-streams JSONL records in a date range,
    buckets them by 3-hour Kp window, and emits a Markdown table
    of proxy + scintillation metrics per bucket plus a Kp-severity
    aggregate.
  - ``tasks/analysis/2026-05-21_kp_correlation.md`` — first run
    over May 14-21 (8 days, ~200k records spanning the May 15-16
    G2 storm at Kp=6.33 and the v0.5+ scintillation rollout).

Key findings (full prose in the report file):

  1. σ_φ thresholds still too low: at Kp=1.00 (quietest bucket)
     σ_φ averages 1.27 rad with 77% of peaks "strong" under v0.5.2.
     HF intrinsic floor at SEAB sits at ~1.2-1.5 rad.  Drove the
     v0.6.2 recalibration below.
  2. F2_extreme rate is suspiciously high (~35%) and uncorrelated
     with Kp.  Hypothesis: ``invert()`` mishandles 3-hop returns
     (group_range ≈ 2000+ km at SEAB) as 1-hop with implausibly
     high h'.  Worth a dedicated investigation in a later release.
  3. The G2 storm did nudge metrics, but the SNR drop and CPI-count
     drop are more reliable disturbance signals than F2_extreme
     rate.


## v0.6.2 — σ_φ Kp-calibrated thresholds (2026-05-21)

Direct outcome of finding 1 above.  v0.5.2's recalibration (0.5/1.0)
was correct in direction but insufficient in magnitude; the
Kp-correlation analysis shows the HF intrinsic floor at this path
sits around 1.2-1.5 rad even on the quietest geomagnetic days.

### Tasks

- [x] ``core/scintillation.py``: bump ``SIGMA_PHI_WEAK_MAX`` 0.5 →
      1.5, ``SIGMA_PHI_MODERATE_MAX`` 1.0 → 2.0,
      ``SIGMA_PHI_EVENT_THRESHOLD`` 0.5 → 1.5.  Update module
      docstring with the calibration history table.
- [x] Tests:
      - ``TestSigmaPhiSeverityHelper``: update the literal
        out-of-bin probe value 1.5 → 2.5 (1.5 is now the
        weak/moderate boundary, not "strong" territory).
      - ``TestSigmaPhiSeverityEndToEnd``: restrict parametrization
        to the unwrap-safe regime (target σ ≲ 1.0).  At higher
        targets the orthogonal-quadratic phase pattern's per-sample
        step exceeds π → ``np.unwrap`` interferes with the
        constructed signal.  Helper tests already cover boundary
        assignment at exact float64 values where unwrap is
        irrelevant.
      - ``test_event_when_sigma_phi_at_threshold``: rewrite to use
        random-uniform phases (σ ≈ π/√3 ≈ 1.81 > 1.5 event
        threshold) instead of the orthogonal-quad pattern which
        couldn't reach 1.5 σ safely.
- [x] ``README.md`` v0.6.2 highlights with calibration rationale.
- [x] ``pyproject.toml`` + ``deploy.toml`` — version 0.6.1 → 0.6.2.

### Out of scope (deferred)

- **F2_extreme misclassification investigation** — separate
  effort; needs a probe that loads F2_extreme records and tests
  multi-hop re-interpretations against typical F2 heights.
- **Storm-day σ_φ recalibration** — wait for a Kp ≥ 5 event with
  v0.5+ logging.  Re-run ``kp_correlation_analysis.py`` once that
  data exists.

### Verification status

212 tests pass (was 214 in v0.6.1; -2 net because the e2e tests
shrank from 6 parametrize cases to 4 unwrap-safe ones, while the
helper boundary tests gained equivalent coverage).  Live
verification on bee1-rx888: σ_φ_severity flipped 100% to weak
(was ~85% strong under v0.6.1); event rate STILL 94% because S4
was now driving events alone — leading immediately to v0.6.3.


## v0.6.3 — S4 thresholds Kp-calibrated (2026-05-21)

Same Kp-correlation finding applied to S4: at HF oblique with
multipath the signal Rayleigh-fades and produces S4 ≈ 0.7-1.0 by
construction.  Quiet-day distribution (May 21, Kp 1.0-3.0, 11,577
records): median S4 = 0.78, p90 = 1.05 — ITU-R's 0.3/0.6
thresholds classify a typical quiet peak as "strong scintillation".

### Tasks

- [x] ``core/scintillation.py``: bump ``S4_WEAK_MAX`` 0.3 → 1.0,
      ``S4_MODERATE_MAX`` 0.6 → 1.5, ``S4_EVENT_THRESHOLD`` 0.3 →
      1.0.  Update module docstring with the calibration history
      table covering both indices.  Update README v0.6.3 highlights.
- [x] Tests:
      - ``TestS4SeverityHelper``: replace literal probe value 0.9
        ("strong" under ITU-R) → 2.0 (still "strong" under v0.6.3).
      - ``TestS4SeverityEndToEnd``: replace ``_BOUNDARY_TOL``
        parametrization with well-within-bin probe values.  The
        saturated-branch construction of
        ``_amplitudes_with_target_s4`` has integer-rounding error
        from ``n_high = round(n/(1+S4²))`` that makes boundary-
        adjacent tests fragile; helper tests cover exact
        boundaries.
      - ``_amplitudes_with_target_s4``: extend to support
        target_s4 > 1.0 via a two-level (high/zero) construction
        with ``n_high : n_low = 1 : target_s4²``.
      - ``test_event_when_s4_above_threshold``: rewrite with target
        s4 = 1.2 (safely above 1.0 event threshold).  Add explicit
        ``s4_index >= S4_EVENT_THRESHOLD`` assertion before the
        event assertion.
- [x] ``pyproject.toml`` + ``deploy.toml`` — version 0.6.2 → 0.6.3.
      ``contract_version`` stays 0.6 (no contract surface change).

### Out of scope (still / again)

- **F2_extreme misclassification investigation** — separate work;
  needs an analytical probe to test multi-hop interpretation of
  long-group-range peaks.  Now the most pressing followup since
  σ_φ + S4 are both calibrated.
- **Storm-day final calibration** — both σ_φ and S4 thresholds
  remain provisional until a Kp ≥ 5 event with v0.5+ logging.
  Re-running ``kp_correlation_analysis.py`` post-storm will
  refresh.

### Verification status

211 tests pass (was 212 in v0.6.2; -1 net from dropping the
saturated-construction-incompatible test point).  Live
verification on bee1-rx888: post-deploy data shows S4 100% weak,
σ_φ ~62/19/19 weak/moderate/strong, event rate 38% (was 94% under
v0.6.2) — events now driven by genuine σ_φ excursions above the
HF intrinsic floor rather than baseline Rayleigh fading.


## v0.7.0 — multi-hop inversion (2026-05-21)

Closes the F2_extreme misclassification finding from the Kp
correlation analysis.  Diagnostic
(``tasks/analysis/2026-05-21_f2_extreme_multihop_diagnostic.md``)
showed 100% of F2_extreme records have a clean 3-hop
interpretation at typical F2 heights (mean h' = 263 km),
confirming that the v0.5/0.6 1-hop-only inversion was
systematically mislabelling multi-hop returns as F2_extreme at
implausibly high altitudes.

### Tasks

- [x] ``core/invert.py``:
      - Generalised ``virtual_height_km(P, D, n_hops=1)`` to the
        multi-hop form ``h = sqrt(P²-D²) / (2N)``.
      - Generalised ``virtual_height_uncertainty_km`` with the
        ``N²`` denominator factor from differentiating the
        multi-hop expression.
      - ``equivalent_vertical_freq_mhz`` and ``takeoff_zenith_deg``
        unchanged (both ``N``-invariant from geometry).
      - New ``select_n_hops(P, D, max_hops=4)`` picks the most
        plausible ``N``: keep ``N = 1`` when ``h_1 < 500 km``
        (typical E/F1/F2); otherwise search ``N ≥ 2`` for the
        smallest giving plausible F-region height; fall back to
        ``N = 1`` if none qualify.
      - ``IonosphericFix`` gains ``n_hops: int`` field.
      - ``invert()`` auto-selects ``N`` (override via ``n_hops``
        kwarg for tests / external disambiguation).
- [x] Wire schema: ``n_hops`` added to JSONL records and
      ``codar.spots`` sink rows.  Daemon log line shows ``2F``,
      ``3F`` etc. next to ``mode_layer`` for at-a-glance visibility.
- [x] ``scripts/multihop_diagnostic.py`` (new) — re-runnable
      diagnostic that loads F2_extreme records, computes
      ``h'_apparent`` under N = 1, 2, 3, 4 hops, and reports the
      climatological fit.
- [x] ``tasks/analysis/2026-05-21_f2_extreme_multihop_diagnostic.md``
      — first-run report.
- [x] Tests:
      - ``TestVirtualHeightMultiHop`` (5 tests): N=1 matches
        default; N=2/3 give expected divided heights; SEAB
        diagnostic example reproduces the 3-hop interpretation;
        invalid n_hops raises.
      - ``TestVirtualHeightUncertaintyMultiHop`` (2 tests): N=1
        matches default; uncertainty scales as 1/N at the same
        h_1 geometry.
      - ``TestSelectNHops`` (8 tests): typical F2/F1/E pick N=1;
        SEAB diagnostic example picks N=2 or N=3 by hop magnitude;
        fallback when no plausible interpretation; geometry
        violation raises.
      - ``TestInvertMultiHop`` (6 tests): auto-selection for
        typical case = N=1; F2_extreme apparent → N=2 +
        reclassified F2; explicit override; fv and takeoff
        zenith N-invariant; uncertainty smaller for multi-hop.
- [x] ``test_multi_peak.py``: ``expected_cols`` set extended with
      ``n_hops``; integration test asserts the field is int ≥ 1.
- [x] ``README.md`` v0.7.0 highlights with semantic-change caveat.
- [x] ``pyproject.toml`` + ``deploy.toml`` — version 0.6.3 → 0.7.0.
      ``contract_version`` stays 0.6 (no contract surface change).

### Out of scope (deferred)

- **Multi-hop disambiguation via SNR climatology** — when both N=2
  and N=3 give plausible heights, the v0.7 selector prefers N=2
  (smaller).  A SNR-based preference (3-hop returns are ~10-20 dB
  weaker than 2-hop on the same path) would be physically
  motivated but requires absolute-power calibration we don't have.
- **Cross-CPI rolling-window scintillation** — original v0.5
  deferred item; underfit_ratio from v0.6.0 reduces its urgency.

### Verification status

232 tests pass (was 211 in v0.6.3; +21 new for multi-hop coverage).
24 pre-existing Kaeppler Zenodo-dataset skips unchanged.  Live
verification on bee1-rx888: TBD post-deploy — expect
mode_layer distribution to shift from ~37% F2 / 37% F2_extreme to
~67% F2 / nearly 0% F2_extreme (the former being a sum of 1-hop
and reclassified 2-hop and 3-hop returns).

# codar-sounder

Opportunistic ionospheric sounder for the HamSCI sigmond suite.  Receives
CODAR (Coastal Ocean Dynamics Applications Radar) HF chirp transmissions
via radiod, dechirps them, and produces a JSON-Lines time series of group
range, virtual height, and equivalent vertical frequency along each
oblique propagation path.

Per Kaeppler et al. (2022, *Atmos. Meas. Tech.* 15:4531-4545).  CODAR
transmitters along the US east and west coasts radiate linear-FMCW chirps
at well-characterised frequencies (4-50 MHz) 24/7; their already-paid-for
GPS-disciplined signals are an excellent opportunistic source for
single-frequency oblique ionospheric sounding.

## Status

**v0.5.0 — adds ITU-R P.531 scintillation indices on top of v0.4's
feature-complete single-antenna release.**  Contract surfaces
(`inventory --json`, `validate --json`, `version --json`,
`config init|edit`, `tdma-scan`) work end-to-end against the sigmond
v0.6 contract.  The daemon (`core/daemon.py`) routes IQ → dechirp →
trace → invert → scintillation → JSONL+CH writer per CPI:

  * **Dechirp** (`core/dechirp.py`) — Kaeppler §2.1: windowed quadratic-
    phase replica, range-Doppler FFT, beat → group-range conversion;
    TDMA phase-offset wrapping for co-band transmitters.
  * **Trace** (`core/trace.py`) — rolling-median ground-clutter mask,
    multi-peak detection (`find_f_region_peaks`) with SNR threshold and
    minimum-separation collapsing.  Up to 4 peaks per CPI surface every
    open propagation mode (1F2 high-ray + low-ray, plus E-layer and
    Es returns when present).
  * **Invert** (`core/invert.py`) — secant-law virtual height +
    equivalent vertical frequency with Kaeppler Eq. 13/14 uncertainty
    propagation.  Each fix is layer-classified (`E`/`F1`/`F2`/
    `F2_extreme`/`below_E`) by virtual height — see `classify_layer`.
  * **Output** (`core/output.py`) — daily-rotated JSONL at
    `/var/lib/codar-sounder/<radiod>/<station>/YYYY/MM/DD.jsonl`,
    one record per detected peak with `peak_index` / `peak_count` /
    `mode_layer`.  Remains the canonical L1 artefact (Kaeppler-
    compatible Zenodo schema).
  * **Scintillation** (`core/scintillation.py`, v0.5+) — per peak per
    CPI, the pre-Doppler-FFT range-bin slow-time vector is reduced to
    ITU-R P.531 S4 (amplitude) and σ_φ (phase) indices with severity
    bins (weak / moderate / strong / unknown).  A propagation-mode-
    resolved companion to hf-timestd's vertical-incidence WWV
    scintillation: oblique geometry, mode-by-mode.
  * **HamSCI sink** (CONTRACT v0.6 §17) — when the local HamSCI sink
    (a SQLite store-and-forward queue managed by sigmond) is in play,
    every per-peak record is also written to `codar.spots` via
    `sigmond.hamsci_sink.Writer`.  The sink path is additive; hosts
    without it stay file-only with no extra moving parts.

**v0.5.0 highlights:**

  * **S4 / σ_φ per peak per CPI** — eight new fields on every JSONL
    record and `codar.spots` row: `s4_index`, `s4_severity`,
    `sigma_phi_rad`, `sigma_phi_severity`, `scintillation_event`,
    `scintillation_confidence`, `scintillation_samples`,
    `mode_doppler_hz`.  Computed from the M-sample slow-time complex
    amplitude vector at each peak's range bin (M = `coherent_seconds`
    × `sweep_repetition_hz`; default M = 60).
  * **Severity bins** (strict-less-than): S4 < 0.3 weak / < 0.6
    moderate / ≥ 0.6 strong (ITU-R P.531 canonical); σ_φ < 0.5 weak /
    < 1.0 moderate / ≥ 1.0 strong (v0.5.2 HF-recalibrated — ITU-R's
    0.2/0.5 thresholds were calibrated for single-mode narrowband
    GNSS/SHF signals, but HF oblique multipath has an intrinsic
    phase-incoherence floor of ~0.4-0.6 rad even on quiet days).
    S4 > 1.0 (saturated scintillation) is *not* clipped.  Event gate
    fires when S4 ≥ 0.3 or σ_φ ≥ 0.5.
  * **Cadence caveat** — at default CPI = 60 s, SRF = 1 Hz, the linear
    detrend acts as an effective ≈ 0.017 Hz high-pass (= 1/CPI),
    *not* the canonical ITU-R 0.1 Hz used by GNSS receivers.  Cross-
    comparisons to GNSS σ_φ should account for this.  The chosen
    linear (not quadratic) detrend is correct at 60 s: TIDs at
    5-60 min period are well-approximated as linear, while genuine
    30-60 s scintillation survives unscathed.
  * **Confidence model**: `min(1, n_samples / 30)` with a NaN/Inf
    guard.  Below 10 slow-time samples (or a clutter-mask null with
    `mean_intensity < 1e-30`) the indices are zeroed and severities
    are `"unknown"`.  Deliberately *not* penalised by S4-correlated
    coefficient-of-variation — that would suppress confidence on real
    strong events.
  * **No contract bump** — `CONTRACT_VERSION` stays 0.6; the new
    fields are additive payload-schema evolution, not contract-shape
    change.

**v0.6.3 — S4 thresholds Kp-calibrated:**

  * v0.6.2 fixed σ_φ but left S4 at ITU-R canonical 0.3/0.6.  Live
    data immediately showed event rate stuck at 94% because S4 was
    now alone driving events: at HF oblique with multipath, the
    signal Rayleigh-fades, producing S4 ≈ 0.7–1.0 by construction
    with no real scintillation.
  * Quiet-day S4 distribution on May 21 (Kp 1.0–3.0; 11,577
    records): p10=0.56, p50=0.78, p90=1.05, p95=1.30.
  * New thresholds: weak < 1.0 / moderate < 1.5 / strong ≥ 1.5.
    Event gate at S4 ≥ 1.0.  Mirror of the σ_φ recalibration in
    v0.6.2.
  * Both indices now in the same HF-recalibrated frame; absolute
    values remain comparable to other HF multipath sounders but
    NOT directly to GNSS scintillation literature.

**v0.6.2 — σ_φ thresholds Kp-calibrated:**

  * Severity bins re-tuned from a 60-bucket Kp-correlation analysis
    (`tasks/analysis/2026-05-21_kp_correlation.md`).  At Kp = 1.00
    (very quiet) production σ_φ averages 1.27 rad with 77% of peaks
    flagging "strong" under v0.5.2 — proving the v0.5.2 thresholds
    (0.5 / 1.0) sat well below the HF intrinsic phase-incoherence
    floor at SEAB / 13.45 MHz / 1416 km.
  * New thresholds: weak < 1.5 / moderate < 2.0 / strong ≥ 2.0.
    Event gate at σ_φ ≥ 1.5.  S4 thresholds stay ITU-R canonical
    (0.3 / 0.6).
  * Caveat: the 12-hour scintillation-enabled window covered only
    Kp 1.0–3.0 (quiet → unsettled).  Final calibration awaits a
    Kp ≥ 5 storm with v0.5+ logging — expect another small nudge
    if storm-day σ_φ statistics warrant it.

**v0.6.1 — per-sweep MAD pre-filter:**

  * **`dechirp_sweeps_rejected`** wire field per record.  Sweeps whose
    post-fast-time-FFT total power deviates more than 4·MAD from the
    CPI median are zeroed before the slow-time FFT.  Removes sferic-
    like impulses, discrete-tone RFI bursts (e.g. shortwave broadcast
    carriers near the CODAR band), and longer disturbances at the
    sweep level — *before* they corrupt range_profile (peak
    detection) or per-peak slow-time vectors.
  * Live-verification probe identified ~80% of CPIs at 13.45 MHz had
    1-2 anomalous sweeps split across three populations: broadband
    impulses (sferic-like), discrete-tone RFI, and longer-duration
    persistent disturbances.  Per-sweep pre-filter catches all three
    via a single MAD test on per-sweep spectrum total power.
  * Complementary to v0.5.1's per-peak MAD (which catches per-bin
    intensity outliers that survive the pre-filter).  Defense in
    depth.

**v0.6.0 — σ_φ diagnostic fields:**

  * **`sigma_phi_linear_rad` + `sigma_phi_quadratic_rad`** — both
    polyfit-detrend variants exposed as additive wire fields.  The
    canonical `sigma_phi_rad` continues to equal the quadratic value
    (= used for severity classification); the linear variant is the
    σ_φ a v0.5/0.5.1-style consumer would have seen.
  * **`sigma_phi_underfit_ratio`** — `linear / quadratic`.  Equals
    ~1.0 when slow-time phase has no curvature beyond constant
    Doppler (clean single-mode propagation); >> 1 when residual
    curvature exists (TIDs, multipath beating, accelerating
    ionospheric Doppler).  A TID detector independent of σ_φ
    severity classification.
  * 3 additive wire fields; contract version unchanged at 0.6.

**v0.4.0 highlights:**

  * **Wideband filter wiring** (`core/stream.py`) — uses ka9q-python
    ≥3.11's `low_edge` / `high_edge` filter kwargs so the captured IQ
    spans the full chirp bandwidth (default ±sample_rate/2 ∓ 1500 Hz
    guard, matching the hfdl-recorder pattern).  Without this the `iq`
    preset's default ±5 kHz filter truncated the chirp's edges and
    smeared the dechirped range bins.
  * **Multi-peak detection** — replaces v0.3's argmax single-peak
    pickup so high/low-ray F2 returns and concurrent E-layer paths
    surface as distinct records.
  * **Layer classification** — virtual-height bins per Davies (1990)
    digisonde conventions; sporadic-E is folded into the E label
    (cannot be reliably distinguished from regular E without an MUF
    sweep).
  * **`tdma-scan --write-config`** — persists discovered TDMA offsets
    in-place into the config TOML (atomic write, comments preserved).
    Operator runs the scan, eyeballs the SNRs, and re-runs with
    `--write-config` to apply.

**Deferred (waiting on a second antenna):** cross-loop / crossed-
dipole AOA — needs the second physical antenna to extract Stokes
parameters.  Skipped for this release; revisit when antenna lands.

**Permanent non-goals:** dynamic TDMA re-lock (CODAR TXs are GPS-
disciplined; drift is negligible at the timescales we care about) and
HFRNet table import (self-discovery is sufficient and avoids an
external-data dependency).

## Install

Pattern A (sigmond-managed):

```
sudo smd install codar-sounder
sudo smd apply        # writes the [[radiod.fragment]] channel into radiod
sudo systemctl start codar-sounder@<radiod-id>
```

Standalone (without sigmond):

```
sudo ./scripts/install.sh
sudo systemctl start codar-sounder@<radiod-id>
```

See [docs/CONTRACT.md] in the sigmond repo for the v0.5 contract.

## License

MIT.

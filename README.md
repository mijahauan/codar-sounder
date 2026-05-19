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

**v0.4.0 — feature-complete single-antenna release.**  Contract
surfaces (`inventory --json`, `validate --json`, `version --json`,
`config init|edit`, `tdma-scan`) work end-to-end against the sigmond
v0.6 contract.  The daemon (`core/daemon.py`) routes IQ → dechirp →
trace → invert → JSONL+CH writer per CPI:

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
  * **HamSCI sink** (CONTRACT v0.6 §17) — when the local HamSCI sink
    (a SQLite store-and-forward queue managed by sigmond) is in play,
    every per-peak record is also written to `codar.spots` via
    `sigmond.hamsci_sink.Writer`.  The sink path is additive; hosts
    without it stay file-only with no extra moving parts.

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

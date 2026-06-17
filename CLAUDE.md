# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**codar-sounder** is an opportunistic ionospheric sounder for the
HamSCI (Ham Radio Science Citizen Investigation) sigmond suite. It
receives CODAR (Coastal Ocean Dynamics Applications Radar)
high-frequency (HF) linear frequency-modulated continuous-wave
(FMCW) chirp transmissions via `radiod`, dechirps them, and
produces a JSON Lines (JSONL) time series of group range, virtual
height, and equivalent vertical frequency along each oblique
propagation path.

The science follows Kaeppler et al. (2022, *Atmos. Meas. Tech.* 15:
4531–4545). CODAR transmitters along the US east and west coasts
radiate linear-FMCW chirps at well-characterised frequencies
(4–50 MHz) 24/7 with timing disciplined by the Global Positioning
System (GPS) — already-paid-for signals that are an excellent
opportunistic source for single-frequency oblique sounding.

Part of the HamSCI sigmond suite — see `/opt/git/sigmond/sigmond/CLAUDE.md`
(orchestrator) and `/opt/git/sigmond/CLAUDE.md` (umbrella) for
cross-repo context. README carries background and operator-facing
overview; `docs/METHODOLOGY.md` carries the signal-processing /
inversion / scintillation detail; this file is the operational map.

## Authors

- Michael Hauan (AC0G, GitHub: mijahauan)
- Repo: https://github.com/mijahauan/codar-sounder

## Quick reference

```bash
# Development — uv canonical
uv sync --extra dev
uv run pytest tests/
uv run pytest tests/test_dechirp.py -v           # one file
uv run pytest -k scintillation -v                # by keyword

# Production install / upgrade (uses sigmond's shared _ensure_uv helper;
# auto-clones ka9q-python sibling if missing at /opt/git/sigmond/ka9q-python)
sudo ./scripts/install.sh           # first-run
sudo ./scripts/deploy.sh            # ongoing refresh

# Command-line interface (CLI) — verify against `codar-sounder --help`
codar-sounder inventory --json      # per-instance resource view
codar-sounder validate --json       # config validation
codar-sounder version --json
codar-sounder daemon --config <path> --radiod-id <id>
codar-sounder tdma-scan             # probe time-division multiple access (TDMA) slot assignments for co-band tx's
codar-sounder config init|edit      # whiptail wizard via sigmond.wizard_dispatch
```

## Architecture

```
radiod (ka9q-radio, IQ preset)
  │   wideband in-phase / quadrature (I/Q) for the CODAR chirp band —
  │   ka9q-python's ensure_channel(low_edge=..., high_edge=...) bypasses
  │   the iq preset's ±5 kHz audio filter that would clip the chirp.
  ▼
RadiodIQSource (core/stream.py)
  │   complex 32-bit float (CF32) I/Q samples → CPI (coherent
  │   processing interval) framing
  ▼
Dechirp (core/dechirp.py)                       — Kaeppler §2.1
  │   windowed quadratic-phase replica, range-Doppler fast Fourier
  │   transform (FFT), beat-frequency → group-range conversion
  │   TDMA phase-offset wrapping for co-band transmitters
  ▼
Trace (core/trace.py)
  │   rolling-median ground-clutter mask
  │   find_f_region_peaks() — multi-peak detection (signal-to-noise
  │   ratio (SNR) threshold, min separation, ≤4 peaks per CPI:
  │   1F2 high+low ray, E, sporadic-E (Es))
  ▼
Invert (core/invert.py)
  │   secant-law virtual height + equiv vertical freq
  │   Kaeppler Eq. 13/14 uncertainty propagation
  │   classify_layer() → E / F1 / F2 / F2_extreme / below_E
  ▼
Scintillation (core/scintillation.py, v0.5+)
  │   per peak per CPI: pre-Doppler-FFT range-bin slow-time
  │   → International Telecommunication Union Radiocommunication
  │   Sector (ITU-R) Recommendation P.531 S4 (amplitude) + σ_φ
  │   (phase) indices
  │   severity bins (weak / moderate / strong / unknown)
  │   propagation-mode-resolved (mode-by-mode vs hf-timestd's
  │   vertical-incidence approach using the WWV time-and-frequency
  │   broadcast from the US National Institute of Standards and
  │   Technology (NIST))
  ▼
Output (core/output.py)
  │   Canonical Level-1 (L1) artefact (Kaeppler-compatible Zenodo schema):
  │   /var/lib/codar-sounder/<radiod>/<station>/YYYY/MM/DD.jsonl
  │   One record per detected peak (peak_index / peak_count / mode_layer).
  ▼
HamSCI sink — additive (CONTRACT §17)
  │   codar.spots rows via sigmond.hamsci_sink.Writer when
  │   /var/lib/sigmond/sink.db is writable; no-op otherwise.
```

## Project structure

```
src/codar_sounder/
  cli.py                  # argparse entry — subcommands listed above
  config.py               # TOML loader, defaults
  configurator.py         # `config init|edit` whiptail via sigmond.wizard_dispatch
  contract.py             # inventory/validate JSON builders (CONTRACT_VERSION = "0.7")
  tdma_config_writer.py   # render TDMA slot config from tdma-scan results
  version.py              # GIT_INFO dict
  core/
    daemon.py             # orchestrates the per-CPI pipeline below
    stream.py             # RadiodIQSource — wideband IQ via ka9q-python
                          #   (uses low_edge/high_edge kwargs added in ≥3.11)
    dechirp.py            # Kaeppler §2.1 dechirp + range-Doppler FFT
    trace.py              # ground-clutter mask + multi-peak detection
    invert.py             # virtual height + equiv vertical freq + uncertainties
    scintillation.py      # S4 + σ_φ per peak per CPI
    output.py             # daily-rotated JSONL + HamSCI sink writer
    tdma.py               # TDMA slot detection / phase wrapping
    authority_reader.py   # §18 timing-authority snapshot subscriber
data/codar-stations.toml  # CODAR transmitter site database (frequencies, locations)
tests/                    # 11 files; dechirp / trace / invert / scintillation / contract / TDMA
config/                   # codar-sounder-config.toml.template
etc/                      # additional config fragments
scripts/                  # install.sh, deploy.sh, kp_correlation_analysis.py,
                          # multihop_diagnostic.py (calibration & monitoring tools)
systemd/                  # codar-sounder@.service template unit
deploy.toml               # sigmond client manifest
tasks/                    # planning notes
```

## Key design decisions

- **One systemd instance per radiod**
  (`codar-sounder@<radiod_id>.service`), matching the other recorders.
- **Wideband IQ, not the iq-preset audio filter.** `RadiodIQSource`
  passes `low_edge=` / `high_edge=` to `ensure_channel()` (ka9q-python
  ≥3.11) so the CODAR chirp edges aren't truncated by the iq preset's
  ±5 kHz default audio filter. The repo declares
  `ka9q-python>=3.14.0` (for `client_id=` per-(client,radiod)
  multicast routing, contract §7).
- **JSONL is the canonical L1 artefact.** Daily-rotated, one record
  per detected peak. The HamSCI SQLite sink path is *additive* — it
  augments rather than replaces JSONL, and silently no-ops when the
  sink is absent. This preserves Zenodo-compatible schema and means
  standalone hosts work without sigmond's sink.
- **Multi-hop inversion (v0.7).** Up to 4 peaks per CPI surface each
  open propagation mode (1F2 high+low ray, E, Es). `mode_layer`
  classification routes each fix to the right layer based on
  virtual height; closes the F2_extreme misclassification mode of
  earlier versions.
- **Kp-calibrated scintillation thresholds.** S4 and σ_φ severity
  bins (weak / moderate / strong) were calibrated against the
  planetary geomagnetic activity index (Kp) in v0.6.2/0.6.3 —
  see `scripts/kp_correlation_analysis.py` and
  `scripts/multihop_diagnostic.py` for the analysis. Kp ≥ 5 storm-day
  calibration is the open follow-up noted in README.
- **TDMA aware.** Co-located CODAR transmitters share frequencies via
  time-division slots; `core/tdma.py` does phase-offset wrapping so
  the dechirped signal from each site is recovered correctly. Slot
  assignments are discovered via `codar-sounder tdma-scan` and
  written by `tdma_config_writer.py`.

## Client contract (v0.7)

codar-sounder declares `CONTRACT_VERSION = "0.7"` in
`src/codar_sounder/contract.py`. The CLI helpstrings still reference
"v0.5" — that's stale and will mislead `--help` readers; the actual
implementation is at v0.7. Authoritative spec:
`/opt/git/sigmond/sigmond/docs/CLIENT-CONTRACT.md`.

Sections implemented:

- **§1 / §2 / §3 / §4 / §5** — native TOML config, radiod-id binding,
  self-describe CLI, templated systemd, `deploy.toml` manifest.
- **§6 / §7** — uses ka9q-python (`>=3.14.0` for `client_id=` so the
  multicast destination is derived per (client, radiod)); never
  client-specified.
- **§8** — `RADIOD_<id>_CHAIN_DELAY_NS` read from `coordination.env`.
- **§10 / §11** — `log_paths` in inventory; daemon process log goes
  to the systemd journal. `CODAR_SOUNDER_LOG_LEVEL` /
  `CLIENT_LOG_LEVEL` honored on startup and SIGHUP.
- **§12** — validate hardening.
- **§14** — `configurator.py` whiptail wizard via
  `sigmond.wizard_dispatch`.
- **§17** — additive HamSCI sink writer (`codar.spots`) alongside
  canonical JSONL.
- **§18 (timing authority)** — `authority_reader.py` subscribes to
  `/run/hf-timestd/authority.json` and **is** wired into the CPI loop:
  `stream.py:_compute_anchor_utc` anchors each CPI's UTC from
  `rtp_to_wallclock` + the published offset (METROLOGY §4.5), and
  `output.py` records the provenance block per CPI.  The inventory
  `timing_authority_applied` field stays `null` **by design** — `null`
  reports RTP-default mode (RTP-derived label + opportunistic offset),
  the same convention `wspr`/`psk` use; it becomes a populated
  `{source,tier,sigma_ns,...}` object only when a future iteration uses a
  §18 authority to *gate* behavior (e.g. TX-cycle gating), not merely to
  offset labels.

## Production paths

- Config: `/etc/codar-sounder/codar-sounder-config.toml`
- L1 artefact: `/var/lib/codar-sounder/<radiod_id>/<station>/YYYY/MM/DD.jsonl`
- Sigmond sink: `/var/lib/sigmond/sink.db` (additive `codar.spots`)
- Process log: systemd journal (`journalctl -u codar-sounder@<radiod_id>`)
- Venv: `/opt/codar-sounder/venv`
- Source: `/opt/git/sigmond/codar-sounder` (editable install)
- Service user: `codarsnd:codarsnd`

## Calibration & monitoring tooling

`scripts/` carries two analysis utilities the daemon doesn't depend
on but which inform threshold calibration and multi-hop debugging:

- `kp_correlation_analysis.py` — correlate scintillation severity
  vs Kp index (used to set S4 / σ_φ thresholds in v0.6.2/0.6.3).
- `multihop_diagnostic.py` — visualise per-CPI peak structure to
  validate multi-hop classification (used in v0.7).

See `docs/METHODOLOGY.md` §11 for usage and the open Kp ≥ 5
storm-day follow-up.

## Reference

- Kaeppler, S. R. et al. (2022). "Demonstration of opportunistic
  ionospheric sounding using CODAR transmissions in the United States."
  *Atmos. Meas. Tech.* 15, 4531–4545. doi:10.5194/amt-15-4531-2022.
  This paper drives §2.1 (dechirp), Eq. 13/14 (uncertainty
  propagation), and the data product schema.
- `README.md` — background, overview, install.
- `docs/METHODOLOGY.md` — signal-processing methodology, formulas,
  thresholds, release-by-release evolution.

## Per-instance cutover (Phase 5 of sigmond multi-instance architecture)

The systemd unit (`codar-sounder@%i.service`) now passes
`--instance %i` to the daemon (alongside the existing `--radiod-id %i`).
`config.resolve_config_path()` prefers `/etc/codar-sounder/<instance>.toml`
when it exists; otherwise falls back to the legacy shared
`codar-sounder-config.toml` with a one-line `DeprecationWarning`.

Until operators run `sudo smd instance migrate` (sigmond Phase 8),
the per-instance config doesn't exist and existing deployments
keep running unchanged.  After migration, the per-instance config
holds an `[instance]` block with `reporter_id = "AC0G-CODAR"` (or
similar reporter-keyed name); the daemon stops emitting the
deprecation warning.

Spot rows now carry a first-class `reporter_id` field — derived
from the per-instance config's `[instance]` block when present,
falling back to `radiod_id` (matching the existing `instance`
field's semantic) for legacy single-instance deployments.

See `/opt/git/sigmond/sigmond/docs/MULTI-INSTANCE-ARCHITECTURE.md`
for the architecture and phase plan.

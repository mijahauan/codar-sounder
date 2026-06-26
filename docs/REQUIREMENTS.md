# codar-sounder — Requirements Specification

**Status:** v0.1 baseline (retroactive). **Owner:** Michael Hauan (AC0G).
**Last reconciled against code:** codar-sounder `0.7.0` / deploy `0.7.0` / contract `0.8` (git `17f10d6`, 2026-06-25).
**Prefix:** `CDR`.

> Retroactive application of [sigmond/docs/REQUIREMENTS-TEMPLATE.md](https://github.com/HamSCI/sigmond/blob/main/docs/REQUIREMENTS-TEMPLATE.md)
> to an **Active**, science-mature client (v0.7 inversion pipeline, feature-complete
> single-antenna). The sigmond↔component **interface** requirements are specified
> once in the [client contract](https://github.com/HamSCI/sigmond/blob/main/docs/CLIENT-CONTRACT.md)
> (v0.8) and referenced — not restated — here (§8.3). Provenance tags:
> `[DOC]` documented · `[CODE]` implicit-in-code · `[NEW]` surfaced by this review.
> Status: ✅ implemented · 🟡 partial/unverified · ⬜ planned.

## 1. Context & problem statement

CODAR (Coastal Ocean Dynamics Applications Radar) transmitters along the US east
and west coasts radiate **linear frequency-modulated continuous-wave (FMCW)**
chirps at well-characterised frequencies (~4–50 MHz) 24/7, GPS-disciplined. They
exist to image ocean surface currents, but the same signals are an excellent
**opportunistic** source for single-frequency *oblique* ionospheric sounding: the
transmit power is already paid for, the sweep parameters (start frequency,
bandwidth, slope, direction) are public and stable per site, and the coverage
geometry is fixed. codar-sounder receives these chirps through `radiod`,
dechirps them against a matched replica, and recovers ionospheric path
information — group range, virtual height, equivalent vertical frequency,
multi-hop count, and per-peak scintillation — from transmitters it has no
operational relationship with.

The instrument follows Kaeppler et al. (2022, *Atmos. Meas. Tech.*
15:4531–4545), adding multi-hop hypothesis selection, two stages of
median-absolute-deviation (MAD) interference rejection, and per-peak amplitude
(S4) / phase (σ_φ) scintillation indices with HF-recalibrated severity bins. Its
defining principle: the **JSONL spool is the canonical, Kaeppler-compatible,
Zenodo-ready L1 artefact**, and the shared sink is purely additive — a host
without sigmond's sink still produces complete science.

A second design pillar is that **transmitter selection is a picker, not
hand-editing**: ~60 known CODAR sites exist, a given receiver hears only some,
and which ones depends on location and ionosphere. `core/stations.py` ranks the
canonical site database by geodesic distance and owns the MHz→Hz units bridge,
so a hand-transcription error (e.g. a sweep rate off by 1000×) cannot reach the
config.

## 2. Goals & objectives

- Recover **oblique ionospheric geometry** (group range, virtual height,
  equivalent vertical frequency, hop count) per detected peak per CPI, with
  Kaeppler Eq. 13/14 uncertainty propagation.
- Produce **per-peak S4 and σ_φ scintillation** indices with HF-recalibrated
  (Kp-anchored) severity bins, propagation-mode-resolved.
- Emit a **Kaeppler-compatible, Zenodo-ready JSONL** L1 artefact as the canonical
  product; mirror it additively to the shared sink when present.
- Make transmitter selection **error-proof** via a distance-ranked picker driven
  by a canonical site database (no hand-typed `[[radiod.transmitter]]` blocks).
- Run usefully **standalone** (radiod + this component) and as a well-behaved
  multi-instance suite client (one daemon per radiod, off radiod cores,
  timing-authority aware).

## 3. Non-goals / out of scope

- **Being a receiver / tuning hardware.** It consumes pre-tuned wideband IQ from
  `radiod`. (Owner: ka9q-radio.)
- **Angle-of-arrival (AOA) / Stokes-parameter processing** — needs a second
  physical antenna (crossed-dipole / cross-loop); deliberately deferred.
- **Producing a timing authority** — it *consumes* hf-timestd's §18 authority for
  per-CPI UTC labelling; it does not produce one.
- **Ocean-current imaging** — the transmitters' actual purpose is irrelevant here;
  codar-sounder only treats them as signals of opportunity.
- **WERA-waveform stations** — the dechirper handles CODAR FMCW only; the site DB
  excludes WERA.
- **Network-level fusion / cross-station science** — that is PSWS/analysis scope.

## 4. Stakeholders & actors

Station operator · `radiod` (ka9q-radio, wideband IQ source, required) ·
`ka9q-python` (≥3.14 for `client_id=` multicast routing, `low_edge`/`high_edge`
channel kwargs) · `hamsci-dsp` (≥0.2, shared WGS-84 geometry) · `hf-timestd`
(§18 timing-authority producer, optional) · the shared SQLite sink
(`/var/lib/sigmond/sink.db`, additive) + downstream science consumers
(HamSCI archive, TID / space-weather post-processors) · sigmond (multi-instance
lifecycle, CPU affinity, radiod-fragment provisioning, status enrichment,
config-init picker dispatch) · the canonical CODAR site database (HFRNet station
table) · Kaeppler et al. (2022) as the methodological reference.

## 5. Assumptions & constraints

- `CDR-C-001` `[DOC]` ✅ `radiod` SHALL provide a **wideband IQ** channel
  (`preset="iq"`, `mode="iq"`) covering the full CODAR sweep bandwidth + margin
  (default `samprate=64000` at 4.5 MHz); the iq preset's ±5 kHz audio filter
  SHALL be bypassed via `ensure_channel(low_edge=, high_edge=)`.
- `CDR-C-002` `[CODE]` ✅ The CPI timestamp SHALL be RTP-derived plus the §18
  authority offset (`stream.py:_compute_anchor_utc`), never the host wall clock
  (METROLOGY §4.5 invariant); synthetic mode MAY use wall clock.
- `CDR-C-003` `[DOC]` ✅ One systemd instance SHALL run **per radiod**
  (`codar-sounder@<radiod_id>`), each reporting *all* the CODAR transmitters it
  dechirps in parallel; a transmitter is a per-record `station_id`, not an
  instance key.
- `CDR-C-004` `[DOC]` ✅ The canonical site DB (`data/codar-stations.toml`,
  CODAR-FMCW only) SHALL be the sole source of per-site sweep parameters; the
  picker generates config from it and owns the MHz→Hz units bridge.
- `CDR-C-005` `[CODE]` ✅ Python ≥3.10 (`tomllib`/`tomli` shim); `numpy` ≥1.24;
  sibling libs (`ka9q-python`, `hamsci-dsp`) SHALL be editable installs so a
  `git pull` propagates without reinstall.
- `CDR-C-006` `[CODE]` ✅ All transmitters under one `[[radiod]]` block SHALL share
  a band (one wideband IQ subscription); co-band transmitters are separated by
  per-TX dechirp parameters and optional TDMA phase offset.

## 6. Functional requirements

### 6.1 Acquisition
- `CDR-F-001` `[DOC]` ✅ SHALL capture wideband complex IQ from radiod via
  `RadiodIQSource` (ka9q-python), framing contiguous **CPIs** (default 60 s,
  `sample_rate_hz` default 64000), with a `SyntheticIQSource` fallback (loud
  warning) for test/dev.
- `CDR-F-002` `[CODE]` ✅ SHALL open **one IQ subscription per radiod** and fan
  each CPI out across every configured `[[radiod.transmitter]]`, each with its
  own dechirp replica and clutter mask.
- `CDR-F-003` `[CODE]` ✅ SHALL set a radiod channel `LIFETIME` (default
  `2 × CPI × 50` frames), refreshed each CPI, so a crashed daemon's channel
  self-destructs; `0`/`None` means explicit-infinite.

### 6.2 Dechirp (Kaeppler §2.1)
- `CDR-F-010` `[DOC]` ✅ SHALL dechirp each CPI with a windowed quadratic-phase
  replica matched to the transmitter's signed sweep rate + SRF, then a
  range-Doppler FFT, converting beat frequency to group range.
- `CDR-F-011` `[DOC]` ✅ SHALL apply a **per-sweep MAD pre-filter** that zeroes
  impulsive-RFI sweeps before the slow-time FFT, exposing a `bad_sweep_mask` and
  per-CPI `n_sweeps_rejected` count.
- `CDR-F-012` `[DOC]` ✅ SHALL support a per-TX **TDMA phase offset**
  (`tdma_offset_samples`) so co-band, same-frequency transmitters are recovered
  separately; default 0 = no phase alignment.

### 6.3 Trace (multi-peak detection)
- `CDR-F-020` `[DOC]` ✅ SHALL apply a rolling-median **ground-clutter mask**
  (per-transmitter) then SNR- and separation-gated multi-peak detection, up to
  `max_peaks` (default 4) per CPI, sorted SNR-descending, within
  `[range_min_km, range_max_km]`.

### 6.4 Inversion
- `CDR-F-030` `[DOC]` ✅ SHALL invert each peak's group range to **secant-law
  virtual height + equivalent vertical frequency** with Kaeppler Eq. 13/14
  uncertainty propagation, rejecting geometrically impossible peaks
  (group range < ground distance) without crashing the CPI.
- `CDR-F-031` `[DOC]` ✅ SHALL classify each fix into a layer
  (`E`/`F1`/`F2`/`F2_extreme`/`below_E`) and select an ionospheric **hop count**
  (`n_hops`, v0.7 multi-hop selector), reclassifying 1-hop-apparent F2_extreme
  returns to plausible 2-/3-hop F2.

### 6.5 Scintillation
- `CDR-F-040` `[DOC]` ✅ SHALL compute per-peak **ITU-R P.531 S4** (amplitude) and
  **σ_φ** (phase) indices from the pre-Doppler-FFT range-bin slow-time vector,
  with a second-stage MAD outlier reject and a linear-vs-quadratic detrend
  underfit ratio (TID indicator).
- `CDR-F-041` `[DOC]` ✅ Severity bins (`weak`/`moderate`/`strong`/`unknown`) SHALL
  use **HF-recalibrated** thresholds (Kp-anchored, v0.6.2/0.6.3) rather than the
  canonical GNSS ITU-R values; numeric indices SHALL be written full-precision so
  consumers can reproduce the bins deterministically.
- `CDR-F-042` `[NEW]` 🟡 Threshold calibration SHALL cover **Kp ≥ 5 storm days**;
  current bins are calibrated on quiet/moderate data only. *(gap — open
  follow-up noted in README/METHODOLOGY §11.)*

### 6.6 Output
- `CDR-F-050` `[DOC]` ✅ SHALL write daily-rotated JSONL to
  `/var/lib/codar-sounder/<radiod_id>/<station>/YYYY/MM/DD.jsonl`, **one record
  per detected peak** (`peak_index`/`peak_count`/`mode_layer`), rotated at UTC
  midnight; this JSONL is the canonical L1 artefact.
- `CDR-F-051` `[CODE]` ✅ Each record SHALL carry a `timing_authority` provenance
  block (from `authority_reader.read()`, else `standalone_timing_authority`).
- `CDR-F-052` `[DOC]` ✅ SHALL additively write each per-peak row to the shared
  sink table `codar.spots` (`mode="codar"`, `schema_version=1`) via
  `sigmond.hamsci_sink.Writer.from_env`, **no-op** if the sink/module is absent;
  sink failure SHALL NOT block JSONL.
- `CDR-F-053` `[CODE]` ✅ Each sink row SHALL carry a first-class `reporter_id`
  (from the per-instance `[instance]` block, else `radiod_id` fallback).

### 6.7 Transmitter selection (picker)
- `CDR-F-060` `[DOC]` ✅ `stations [--json]` SHALL emit the known CODAR inventory
  ranked by geodesic distance from the receiver (`audible_transmitters`), tagging
  each with band, bearing, and `in_prime_range` (200–2000 km window).
- `CDR-F-061` `[DOC]` ✅ `config init|edit` SHALL present a **whiptail multi-select
  picker** (via `sigmond.wizard_dispatch`) pre-checking prime-range sites,
  grouping choices by band into one `[[radiod]]` block each, and writing them via
  `config apply`.
- `CDR-F-062` `[CODE]` ✅ The picker SHALL own the MHz→Hz **units bridge**
  (`freq_mhz` → `center_freq_hz`) so generated `[[radiod.transmitter]]` blocks
  cannot carry hand-transcription errors.
- `CDR-F-063` `[CODE]` ✅ `config apply` SHALL serialize nested
  `[[radiod.transmitter]]` arrays-of-tables to valid TOML (so the sigmond Textual
  scalar wizard preserves transmitter blocks while delegating array editing to
  whiptail).

### 6.8 TDMA discovery
- `CDR-F-070` `[DOC]` 🟡 `tdma-scan` SHALL probe TDMA slot assignments for co-band
  transmitters; `tdma_config_writer.py` SHALL render the discovered
  `tdma_offset_samples` into config. *(Operator-run tool; not wired into the
  daemon loop.)*

### 6.9 Self-description (contract surface)
- `CDR-F-080` `[CODE]` ✅ SHALL implement `inventory --json` / `validate --json` /
  `version --json` / `config init|edit|show|apply` / `stations` / `tdma-scan`
  per contract v0.8 (see §8.3) with pure-JSON stdout.
- `CDR-F-081` `[CODE]` ✅ `validate` SHALL **fail** on missing receiver lat/lon, no
  `[[radiod]]` block, a radiod missing `status` or `channel_name`, a radiod with
  no transmitters, or a transmitter missing any `REQUIRED_TX_FIELDS`
  (`id`, `center_freq_hz`, `sweep_rate_hz_per_s`, `sweep_bw_hz`,
  `sweep_repetition_hz`, `tx_lat_deg`, `tx_lon_deg`); and **warn** on empty
  callsign or TX-RX distance <50 km / >2000 km.

## 7. Quality / non-functional requirements

- `CDR-Q-001` `[CODE]` ✅ Per-CPI failures (dechirp error, no peak, low SNR,
  unphysical geometry on one of several peaks, sink unreachable) SHALL be logged
  and swallowed — never crash the service.
- `CDR-Q-002` `[CODE]` ✅ The service SHALL be `Type=notify` with `WatchdogSec=180`
  and `Restart=always` (RestartSec=5, `StartLimitBurst=10`/300 s), refreshing the
  watchdog each CPI.
- `CDR-Q-003` `[CODE]` ✅ The shared-sink path SHALL degrade to a graceful no-op
  when the DB/module is unavailable; JSONL output SHALL be unaffected.
- `CDR-Q-004` `[CODE]` ✅ The daemon SHALL run off radiod's CPU cores
  (`preferred_cores="worker"`, sigmond `AFFINITY_UNITS`) so burst FFT processing
  cannot induce RX888 USB drops; memory bounded (`MemoryMax=512M`,
  `MemorySwapMax=0`, `Nice=5`).
- `CDR-Q-005` `[CODE]` ✅ JSONL writes SHALL flush per record (no per-record
  fsync); at most one record lost on crash — an accepted I/O tradeoff.
- `CDR-Q-006` `[CODE]` ✅ Systemd sandboxing SHALL apply: `ProtectSystem=strict`,
  `ProtectHome=read-only`, `NoNewPrivileges`, `ReadWritePaths` limited to
  `/var/lib/codar-sounder` + `/var/log/codar-sounder`, `ReadOnlyPaths=/etc/codar-sounder`,
  caps limited to `CAP_NET_RAW`/`CAP_NET_BIND_SERVICE`.
- `CDR-Q-007` `[CODE]` ✅ All TX-RX geometry SHALL use `hamsci_dsp.geometry`
  (WGS-84 geodesics); `config.haversine_km` is a thin compatibility wrapper over
  `great_circle_km` (geodesic, ~0.5% off legacy spherical — immaterial to the
  50/2000 km thresholds).
- `CDR-Q-008` `[CODE]` ✅ `CODAR_SOUNDER_LOG_LEVEL` / `CLIENT_LOG_LEVEL` SHALL be
  honored at startup and on SIGHUP (contract §11).

## 8. External interfaces

### 8.1 Inputs *(derived from deploy.toml + config + inventory --json)*
- **RF:** radiod wideband IQ via ka9q-python; `[[radiod.fragment]]` priority 40,
  target `*`, template `etc/radiod-fragment.conf` → rendered to
  `/etc/radio/radiod@<id>.conf.d/40-codar-sounder.conf` on `smd apply`
  (channel `codar-4mhz`, `preset="iq"`, `samprate=64000`, `encoding="s16le"`).
- **Config:** `/etc/codar-sounder/codar-sounder-config.toml` (or per-instance
  `/etc/codar-sounder/<instance>.toml` after `smd instance migrate`). Operator
  MUST set: `[station]` `callsign` / `grid_square` / `receiver_lat` /
  `receiver_lon`; ≥1 `[[radiod]]` (`status` mDNS name + `channel_name`) with
  ≥1 `[[radiod.transmitter]]` (the 7 required fields). Optional: `[processing]`
  `coherent_seconds`(60) / `sample_rate_hz`(64000) / `range_min_km`(200) /
  `range_max_km`(800) / `snr_threshold_db`(5.0) / `radiod_lifetime_frames` /
  `force_synthetic` / `synthetic_target_*`; per-TX `tdma_offset_samples`.
- **Timing:** hf-timestd §18 authority at `/run/hf-timestd/authority.json`
  (optional; see §8.3). **Identity:** `/etc/sigmond/coordination.env`
  (passthrough extras `tx_lat_deg`, `tx_lon_deg`, `sweep_rate_hz_per_s`,
  `sweep_bw_hz`, `sweep_repetition_hz`; per-radiod `RADIOD_<id>_CHAIN_DELAY_NS`,
  §8). **Site DB:** `data/codar-stations.toml` (60 sites).

### 8.2 Outputs *(derived from inventory --json + output.py)*
- **Canonical L1:** daily JSONL at
  `/var/lib/codar-sounder/<radiod_id>/<station>/YYYY/MM/DD.jsonl`, one record per
  peak. `data_sinks[].kind="file"`, `retention_days=365`, `mb_per_day=5` per
  instance. Inventory reports **one `instances[]` entry per transmitter**
  (27 for the current sigma config) under one radiod.
- **Shared sink (additive):** `codar.spots` (`target_db="codar"`, `table="spots"`,
  `schema_version=1`) in `/var/lib/sigmond/sink.db`; row fields include `time`,
  `host_call`, `host_grid`, `radiod_id`, `instance`(legacy), `reporter_id`,
  `processing_version`, `station_id`, `oblique_freq_hz`, `sweep_rate_hz_per_s`,
  `coherent_seconds`, `peak_index`, `peak_count`, `mode_layer`, `snr_db`,
  `group_range_km`, `ground_distance_km`, `virtual_height_km`(+unc),
  `equivalent_vertical_freq_mhz`(+unc), `takeoff_zenith_deg`, `n_hops`,
  `s4_index`/`s4_severity`, `sigma_phi_rad`/`sigma_phi_severity`,
  `scintillation_event`/`_confidence`/`_samples`/`_outliers_rejected`,
  `mode_doppler_hz`, `sigma_phi_linear_rad`, `sigma_phi_quadratic_rad`,
  `sigma_phi_underfit_ratio`, `dechirp_sweeps_rejected`.
- **Logs:** process log `/var/log/codar-sounder/<radiod_id>.log` (+ systemd
  journal); `log_paths` in inventory map `<radiod>/<tx>` → process/products.

### 8.3 Contracts / APIs (reference, not restated)
- `CDR-I-001` `[CODE]` ✅ Conforms to **client contract v0.8** (multi-instance);
  `deploy.toml` declares `templated_units=["codar-sounder@.service"]`,
  `requires=[ka9q-python, ka9q-radio]`, `start_priority=200`, a §15
  `[[radiod.fragment]]`, and `[client_features.watch]`/`receiver_channels` UI
  hooks. `inventory` declares `data_sinks=[file]` per instance,
  `provides_timing_calibration=false`, `uses_timing_calibration=false`. Full
  field semantics: contract §3/§15/§17.
- `CDR-I-002` `[DOC]` 🟡 **Timing-authority consumer (label-only):**
  `authority_reader.py` reads hf-timestd's §18 authority and `stream.py`/`output.py`
  stamp it into every CPI record, falling back to RTP-default
  (`standalone_timing_authority`) when absent. `timing_authority_applied` is
  `null` **by design** — codar-sounder uses the authority to *label*, not yet to
  *gate* behaviour (it is the most likely future hard-deadline / TX-cycle-gating
  client; populating the field is a future iteration). Subscriber obligations are
  the contract's, not restated here.
- `CDR-I-003` `[DOC]` ✅ Channel provisioning is the sigmond §15 seam: codar
  contributes `[[radiod.fragment]]`; sigmond renders it into radiod. Per-(client,
  radiod) multicast routing uses ka9q-python `client_id=` (≥3.14, contract §7).

## 9. Data requirements

JSONL record (canonical, schema "v0.5" in `output.py`) carries the full per-peak
fix + scintillation + diagnostics + `timing_authority` block; the sink row is a
flat projection of the same fields (above). Retention: `retention_days=365`,
~5 MB/day per instance. Reference data: `data/codar-stations.toml` — 60 CODAR
(FMCW) sites with `freq_mhz`, signed `sweep_rate_hz_per_s`, `sweep_bw_hz`,
`sweep_repetition_hz`, `tx_lat_deg`/`tx_lon_deg`, `association`; reconciled
against the HFRNet station table on update. Scintillation numerics are written
full-precision so severity bins are reproducible across boundaries.

## 10. Dependencies & development sequence

**Runtime deps:** `radiod` (required), `ka9q-python ≥3.14` + `hamsci-dsp ≥0.2`
(editable siblings), `numpy ≥1.24`, `tomli` (Py<3.11). Optional: `sigmond.hamsci_sink`
(additive sink), `sigmond.wizard_dispatch` (picker), `hf-timestd` (§18 authority).
Service user `codarsnd:codarsnd`; venv `/opt/git/sigmond/codar-sounder/venv`.

**Development sequence (intended, recovered as requirement):** v0.2 single-TX
dechirp/trace/invert → v0.4 feature-complete single-antenna (multi-TX fan-out,
LIFETIME crash-safe channel) → v0.5 ITU-R scintillation + MAD reject → v0.6
per-sweep MAD pre-filter + linear/quadratic underfit diagnostics → v0.6.2/0.6.3
Kp-calibrated severity bins → **v0.7 (current): multi-hop inversion + HF-calibrated
scintillation**. Picker/stations + units bridge + arrays-of-tables serializer
landed alongside (commit `b495a1a`). Deferred: AOA/Stokes (needs second antenna);
Kp ≥ 5 storm-day calibration; §18 authority *gating* (TX-cycle alignment).

## 11. Acceptance criteria & verification

- Contract conformance → `codar-sounder validate --json` (exit 0, no `fail`) +
  surfaced via `smd status`. Current default config: `{"ok": true, "issues": []}`.
- Science correctness → unit suite (`test_dechirp`, `test_trace`, `test_invert`,
  `test_scintillation`, `test_multi_peak`, `test_lifetime`,
  `test_kaeppler_zenodo`) + `multihop_diagnostic.py` per-CPI structure review.
- Picker/units integrity → `test_stations` (ranking + MHz→Hz bridge),
  `test_config_roundtrip` (arrays-of-tables serialize), `test_tdma_config_writer`.
- Sink/JSONL integrity → record schema stability + graceful no-op when sink
  absent (`test_authority_reader`, `test_contract`).
- Standalone operability → `scripts/install.sh` on a radiod-only host runs the
  daemon (synthetic fallback if ka9q-python absent) with no sigmond present.

## 12. Risks & open questions

- `CDR-F-090` `[NEW]` 🟡 **Version/contract-string drift:** `version --json`
  reports `0.3.1` (a stale baked default) while `pyproject`/`deploy`/inventory
  report `0.7.0`; CLI helpstrings still say "v0.5" and CLAUDE.md/contract.py
  docstrings variously say v0.6/v0.7 while the declared `CONTRACT_VERSION="0.8"`.
  These SHALL be reconciled so `--help` and `version` don't mislead.
- `CDR-F-091` `[NEW]` 🟡 **inventory `deps` understates ka9q-python:** the
  `inventory` deps array hard-codes `ka9q-python>=3.11.0` while `pyproject`/`deploy`
  require `>=3.14.0` (for `client_id=`). The advertised floor SHALL match the
  real requirement.
- `CDR-F-092` `[NEW]` ⬜ **§18 authority read-but-not-gated:** authority is stamped
  for provenance, not applied to gate TX-cycle alignment; `timing_authority_applied`
  stays `null`. A future Doppler/TX-gating iteration must close this and populate
  the field (ties to hf-timestd `HFT-I-002`).
- `CDR-F-093` `[NEW]` ⬜ **Single-band synthetic fallback fidelity:** the synthetic
  IQ source models only the first transmitter's sweep; multi-TX synthetic CPIs are
  not represented. Acceptable for smoke-tests; flagged so it isn't mistaken for
  multi-TX coverage.
- `CDR-F-042` (Kp ≥ 5 storm-day calibration) and `CDR-F-070` (tdma-scan not wired
  into the daemon) are the two known science/operability follow-ups.
- **Transmitter selection is distance-only** (mirrors superdarn `audible_radars`);
  geometry/propagation-weighted ranking is a deliberate future increment.

## 13. Traceability

| Requirement | #18 issue | Verification | PSWS #6 |
|---|---|---|---|
| CDR-F-030/031 (multi-hop inversion) | Clients: codar-sounder | test_invert / test_multi_peak | #6:19 (Doppler/iono API) |
| CDR-F-040/041 (HF-calibrated scintillation) | Clients: codar-sounder | test_scintillation + kp_correlation_analysis | #6:31 (sensor integ.) |
| CDR-F-052 (codar.spots sink) | Clients: codar-sounder | sink schema test / additive no-op | #6:31 |
| CDR-F-061/062 (picker + units bridge) | codar transmitter-selection (b495a1a) | test_stations / test_config_roundtrip | — |
| CDR-I-002 (§18 consumer, label-only) | superdarn/codar: §18 gating | authority stamp test | #6:50 |
| CDR-F-042 (Kp≥5 calibration) | *(new — file)* | storm-day fixture | — |
| CDR-F-090 (version/contract drift) | *(new — file)* | version/help review | — |
| CDR-F-091 (deps floor mismatch) | *(new — file)* | inventory-vs-pyproject check | — |
| CDR-F-092 (§18 gating) | *(new — file)* | TX-cycle gating test | #6:50 |

*New rows (CDR-F-042/090/091/092/093) are this review's surfaced gaps; promote to
the #18 Clients: codar-sounder epic.*

# codar-sounder

Opportunistic ionospheric sounder for the HamSCI (Ham Radio
Science Citizen Investigation) sigmond suite.

`codar-sounder` receives CODAR (Coastal Ocean Dynamics
Applications Radar) high-frequency (HF) chirp transmissions via
`radiod`, dechirps them, and produces a JSON Lines (JSONL) time
series of group range, virtual height, equivalent vertical
frequency, and per-peak amplitude / phase scintillation indices
along each oblique propagation path.

## Background

CODAR transmitters along the US east and west coasts radiate
**linear frequency-modulated continuous-wave (FMCW)** chirps at
well-characterised frequencies (4–50 MHz) 24/7, with timing
disciplined by the Global Positioning System (GPS).  Each chirp
sweeps the transmit frequency linearly across a band of roughly
25–100 kHz and repeats at the sweep repetition frequency (SRF;
typically 1 Hz).  Some sites use up-chirps (instantaneous
frequency rising through the sweep), others down-chirps; the
sweep parameters — start frequency, bandwidth, sweep rate,
direction — are stable and well-documented per site.  Co-located
transmitters that share a band coordinate via time-division
multiple access (TDMA) slots, each transmitter starting its
chirp at a fixed phase within a shared repetition period.

Although the transmitters exist to image ocean surface currents,
the same signals are an excellent **opportunistic** source for
single-frequency oblique ionospheric sounding: the transmit power
is already paid for, the frequencies and chirp parameters are
public, and the coverage geometry is fixed and well-documented.
The FMCW waveform is well suited to dechirp-based ranging — beat
frequency after dechirping is proportional to the group-path
delay — so a receiver with the same chirp parameters and
coherent timing can recover ionospheric path information from a
transmitter it has no operational relationship with.

The instrument design follows Kaeppler et al. (2022, *Atmos.
Meas. Tech.* 15:4531–4545).  We add multi-hop hypothesis
selection, two stages of median-absolute-deviation (MAD)
interference rejection, and per-peak amplitude / phase
scintillation indices on top of the original method; see
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the technical
detail.

## What it produces

Daily-rotated JSONL under
`/var/lib/codar-sounder/<radiod_id>/<station>/YYYY/MM/DD.jsonl` —
one record per detected peak per coherent processing interval
(CPI; default 60 s).  Each record carries:

- inverted ionospheric geometry (group range, virtual height,
  equivalent vertical frequency, layer label, multi-hop count),
  with Kaeppler Eq. 13 / 14 uncertainty propagation,
- per-peak amplitude (S4) and phase (σ_φ) scintillation indices
  with HF-recalibrated severity bins,
- diagnostic fields covering interference rejection and detrend
  underfit (an indicator of travelling ionospheric disturbances —
  TIDs).

The JSONL is Kaeppler-compatible and Zenodo-ready.  On hosts
where sigmond's local sink (an SQLite store-and-forward database)
is enabled, every record is also written to the `codar.spots`
table — additive, with no contract change.

## How it works

```
radiod  →  dechirp  →  trace  →  invert  →  scintillation  →  JSONL + sink
```

- **Dechirp** — windowed quadratic-phase replica plus
  range-Doppler fast Fourier transform (FFT; Kaeppler §2.1);
  per-sweep MAD pre-filter removes impulsive radio-frequency
  interference (RFI) before the slow-time FFT.
- **Trace** — rolling-median ground-clutter mask, then
  signal-to-noise-ratio- (SNR-) and separation-gated multi-peak
  detection (up to 4 peaks per CPI surface every open propagation
  mode).
- **Invert** — secant-law virtual height, equivalent vertical
  frequency, and a v0.7 multi-hop hypothesis selector that
  reclassifies 1-hop-apparent F2_extreme returns to plausible
  2- or 3-hop F2 returns.
- **Scintillation** — S4 and σ_φ computed at each peak's range
  bin, with severity thresholds calibrated against the planetary
  geomagnetic activity index (Kp) on real HF oblique data rather
  than the canonical Global Navigation Satellite System (GNSS)
  values recommended by the International Telecommunication
  Union Radiocommunication Sector (ITU-R).

The full algorithmic detail — formulas, defaults, thresholds,
and the rationale for every recalibration — lives in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Status

v0.7.x — multi-hop inversion + HF-calibrated scintillation on top
of v0.4's feature-complete single-antenna release.  Implements
the sigmond client contract v0.8 (`inventory --json`, `validate
--json`, `version --json`, `stations`, `config init|edit`, `tdma-scan`).

The methodology document records the release-by-release evolution
of the science pipeline (`docs/METHODOLOGY.md` §12).

**Deferred:** cross-loop / crossed-dipole angle-of-arrival (AOA)
processing — needs a second physical antenna to extract Stokes
parameters.  Revisit when the second antenna is installed.

## Choosing which transmitters to record

There are ~60 known CODAR sites; a given receiver hears only some of
them, and which ones depends on your location and the ionosphere.  You
don't hand-write transmitter blocks — use the picker:

```
sudo smd config init codar-sounder        # whiptail multi-select picker
```

It lists the known sites ranked by distance from your receiver
(pre-checking those in the prime 200–2000 km one-hop window), groups
your choices by band into radiod channels, and writes a validated
config.  To preview the ranked inventory without configuring:

```
codar-sounder stations --receiver-lat <lat> --receiver-lon <lon>
codar-sounder stations --receiver-lat <lat> --receiver-lon <lon> --json   # for tooling
```

The site database (`data/codar-stations.toml`) is canonical: the picker
generates the MHz→Hz frequency and sweep parameters from it, so a
hand-typed transcription error can't slip into the config.

## Install

Pattern A (sigmond-managed):

```
sudo smd component install codar-sounder
sudo smd apply        # writes the [[radiod.fragment]] channel into radiod
sudo smd config init codar-sounder    # pick transmitters (see above)
sudo systemctl start codar-sounder@<radiod-id>
```

Standalone (without sigmond):

```
sudo ./scripts/install.sh
sudo systemctl start codar-sounder@<radiod-id>
```

The sigmond client contract is documented at
[`sigmond/docs/CLIENT-CONTRACT.md`](https://github.com/HamSCI/sigmond/blob/main/sigmond/docs/CLIENT-CONTRACT.md)
in the orchestrator repo.

## Further reading

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — full
  signal-processing methodology, formulas, thresholds, and
  release-by-release evolution.
- Kaeppler, S. R. et al. (2022).  "Demonstration of opportunistic
  ionospheric sounding using CODAR transmissions in the United
  States."  *Atmos. Meas. Tech.* 15, 4531–4545.
  [doi:10.5194/amt-15-4531-2022](https://doi.org/10.5194/amt-15-4531-2022).

## License

MIT.

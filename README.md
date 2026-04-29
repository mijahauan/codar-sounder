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

**v0.1 — sigmond-integration scaffold.**  Contract surfaces (`inventory
--json`, `validate --json`, `version --json`, `config init|edit`) work
end-to-end against the sigmond v0.5 contract.  The daemon is currently a
stub that subscribes to its radiod IQ channel and logs lifecycle events;
the FMCW dechirping engine (Kaeppler §2.1) lands in v0.2.

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

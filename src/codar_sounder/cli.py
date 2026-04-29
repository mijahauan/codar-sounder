"""codar-sounder CLI entry point.

Subcommands:
    inventory   — contract v0.5 §3 inventory JSON
    validate    — contract v0.5 §12 config validation
    version     — version + git block
    daemon      — long-running sounder (v0.1: stub)
    config init|edit — first-run wizard / edit (contract §14)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path


def _resolve_log_level() -> int:
    """Per contract v0.5 §11 precedence."""
    for env_key in ("CODAR_SOUNDER_LOG_LEVEL", "CLIENT_LOG_LEVEL"):
        val = os.environ.get(env_key, "").upper().strip()
        if val and hasattr(logging, val):
            return getattr(logging, val)
    return logging.INFO


def _install_sighup_handler() -> None:
    """SIGHUP → re-read log level from env (contract §11)."""
    def _on_sighup(signum, frame):
        level = _resolve_log_level()
        logging.getLogger().setLevel(level)
        logging.getLogger(__name__).info(
            "SIGHUP: log level set to %s", logging.getLevelName(level)
        )
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_sighup)


def main():
    # Contract §4: inventory/validate/version must emit clean JSON.
    # Quiet logging early so no INFO lines leak before argparse runs.
    _contract_quiet = any(
        arg in ("inventory", "validate", "version")
        for arg in sys.argv[1:3]
    )

    root = logging.getLogger()
    root.setLevel(logging.WARNING if _contract_quiet else _resolve_log_level())

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        root.addHandler(handler)

    parser = argparse.ArgumentParser(
        prog="codar-sounder",
        description="Opportunistic ionospheric sounder using CODAR chirps",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    def _common(p):
        p.add_argument("--config", type=Path, default=None,
                       help="Path to codar-sounder-config.toml")
        p.add_argument("--log-level", default=None,
                       help="Override log level (DEBUG/INFO/WARNING/ERROR)")

    p_inv = sub.add_parser("inventory", help="Contract v0.5 §3 inventory")
    p_inv.add_argument("--json", action="store_true", default=True)
    _common(p_inv)

    p_val = sub.add_parser("validate", help="Contract v0.5 §12 validation")
    p_val.add_argument("--json", action="store_true", default=True)
    _common(p_val)

    p_ver = sub.add_parser("version", help="Version + git provenance")
    p_ver.add_argument("--json", action="store_true", default=True)
    _common(p_ver)

    p_dae = sub.add_parser("daemon", help="Run sounder daemon")
    p_dae.add_argument("--radiod-id", default=None,
                       help="ID of the [[radiod]] block to use")
    _common(p_dae)

    p_tdma = sub.add_parser(
        "tdma-scan",
        help="Capture IQ from a radiod and discover TDMA TX offsets in the band",
    )
    p_tdma.add_argument("--radiod-id", default=None,
                        help="ID of the [[radiod]] block to use")
    p_tdma.add_argument("--seconds", type=int, default=10,
                        help="Capture duration before discovery (default 10 s)")
    p_tdma.add_argument("--snr-threshold-db", type=float, default=10.0,
                        help="Minimum SNR for a peak to be reported")
    p_tdma.add_argument("--json", action="store_true",
                        help="Emit JSON instead of human-readable output")
    _common(p_tdma)

    p_cfg = sub.add_parser("config", help="Configure codar-sounder")
    cfg_sub = p_cfg.add_subparsers(dest="config_command")
    p_init = cfg_sub.add_parser("init", help="write fresh config from template")
    p_init.add_argument("--reconfig", action="store_true",
                        help="overwrite existing config")
    p_init.add_argument("--non-interactive", action="store_true",
                        help="use env-var defaults, do not prompt")
    _common(p_init)
    p_edit = cfg_sub.add_parser("edit", help="review/update existing config")
    p_edit.add_argument("--non-interactive", action="store_true")
    p_edit.add_argument("--radiod-id", default=None)
    _common(p_edit)

    args = parser.parse_args()

    if not _contract_quiet and getattr(args, "log_level", None):
        level_name = args.log_level.upper()
        if hasattr(logging, level_name):
            root.setLevel(getattr(logging, level_name))

    if args.command == "inventory":
        _handle_inventory(args)
    elif args.command == "validate":
        _handle_validate(args)
    elif args.command == "version":
        _handle_version(args)
    elif args.command == "daemon":
        _handle_daemon(args)
    elif args.command == "tdma-scan":
        _handle_tdma_scan(args)
    elif args.command == "config":
        _handle_config(args)
    else:
        parser.print_help()
        sys.exit(1)


def _resolved_config_path(args) -> Path:
    return args.config or Path(
        os.environ.get("CODAR_SOUNDER_CONFIG",
                       "/etc/codar-sounder/codar-sounder-config.toml")
    )


def _handle_inventory(args):
    from codar_sounder.config import load_config
    from codar_sounder.contract import CONTRACT_VERSION, build_inventory

    config_path = _resolved_config_path(args)
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        payload = {
            "client": "codar-sounder",
            "version": "0.1.0",
            "contract_version": CONTRACT_VERSION,
            "config_path": str(config_path),
            "instances": [],
            "issues": [{
                "severity": "fail", "instance": "all",
                "message": f"config not found: {config_path}",
            }],
        }
        print(json.dumps(payload, indent=2))
        return
    payload = build_inventory(config, config_path)
    print(json.dumps(payload, indent=2))


def _handle_validate(args):
    from codar_sounder.config import load_config
    from codar_sounder.contract import build_validate

    config_path = _resolved_config_path(args)
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        payload = {
            "ok": False,
            "config_path": str(config_path),
            "issues": [{
                "severity": "fail", "instance": "all",
                "message": f"config not found: {config_path}",
            }],
        }
        print(json.dumps(payload, indent=2))
        sys.exit(1)
        return
    payload = build_validate(config, config_path)
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        sys.exit(1)


def _handle_version(args):
    from codar_sounder import __version__
    from codar_sounder.version import GIT_INFO
    payload = {"client": "codar-sounder", "version": __version__}
    if GIT_INFO:
        payload["git"] = GIT_INFO
    print(json.dumps(payload, indent=2))


def _handle_daemon(args):
    _install_sighup_handler()
    from codar_sounder.config import load_config, resolve_radiod_block
    from codar_sounder.core.daemon import SounderDaemon

    config_path = _resolved_config_path(args)
    config = load_config(config_path)
    block = resolve_radiod_block(config, args.radiod_id)

    log = logging.getLogger("codar_sounder.daemon")
    log.info("starting codar-sounder daemon for radiod %s (config=%s)",
             block.get("id", "default"), config_path)

    daemon = SounderDaemon(config, block)
    daemon.run()


def _handle_tdma_scan(args):
    """Capture IQ for one CPI and run TDMA offset discovery on it.

    Output (human-readable):
        TDMA scan on radiod=bee1-rx888 channel=codar-4mhz κ=-25733.9 Hz/s
        capture: 10 s = 640000 samples at 64000 Hz
        discovered 3 peaks (snr_threshold=10 dB):
          peak 1: offset_samples=  256  snr=42.3 dB
          peak 2: offset_samples=21504  snr=38.1 dB
          peak 3: offset_samples=43008  snr=35.7 dB
        per-TX TDMA offsets (after subtracting ground delay):
          LISL (D=380 km, 81 samples): tdma_offset_samples=175
          ASSA (D=560 km, 119 samples): tdma_offset_samples=21385
          CEDR (D=580 km, 124 samples): tdma_offset_samples=42884

    Operator pastes the per-TX values into the [[radiod.transmitter]]
    blocks of /etc/codar-sounder/codar-sounder-config.toml.
    """
    _install_sighup_handler()
    from codar_sounder.config import (
        haversine_km, load_config, resolve_radiod_block, transmitters,
    )
    from codar_sounder.core.stream import make_iq_source
    from codar_sounder.core.tdma import (
        discover_tx_offsets, match_peaks_to_txs,
    )

    config_path = _resolved_config_path(args)
    config = load_config(config_path)
    block = resolve_radiod_block(config, args.radiod_id)

    txs = list(transmitters(block))
    if not txs:
        sys.stderr.write(
            f"No [[transmitter]] blocks found for radiod "
            f"{block.get('id', '?')}\n"
        )
        sys.exit(2)

    # Open an IQ source against the radiod (or synthetic fallback) for
    # one capture window, then close it.
    sample_rate_hz = float(
        config.get("processing", {}).get("sample_rate_hz", 64000)
    )
    first_tx = txs[0]
    iq_source = make_iq_source(
        radiod_status_dns=str(block.get("status_dns", "")),
        channel_name=str(block.get("channel_name", "codar")),
        sample_rate_hz=sample_rate_hz,
        cpi_seconds=float(args.seconds),
        sweep_rate_hz_per_s=float(first_tx["sweep_rate_hz_per_s"]),
        sweep_repetition_hz=float(first_tx["sweep_repetition_hz"]),
        center_freq_hz=float(first_tx["center_freq_hz"]),
        preset=str(block.get("preset", "iq")),
        fallback_target_group_range_km=500.0,
        force_synthetic=False,
    )

    log = logging.getLogger("codar_sounder.tdma_scan")
    log.info(
        "tdma-scan: radiod=%s channel=%s capture=%d s sample_rate=%d Hz",
        block.get("id", "?"), block.get("channel_name", "?"),
        args.seconds, int(sample_rate_hz),
    )

    try:
        rx_samples = next(iter(iq_source))
    except StopIteration:
        sys.stderr.write("IQ source produced no samples\n")
        sys.exit(3)
    finally:
        if hasattr(iq_source, "stop"):
            iq_source.stop()

    sweep_rate_hz_per_s = float(first_tx["sweep_rate_hz_per_s"])
    sweep_repetition_hz = float(first_tx["sweep_repetition_hz"])
    n_per_sweep = int(round(sample_rate_hz / sweep_repetition_hz))

    peaks = discover_tx_offsets(
        rx_samples,
        sample_rate_hz=sample_rate_hz,
        sweep_rate_hz_per_s=sweep_rate_hz_per_s,
        sweep_repetition_hz=sweep_repetition_hz,
        snr_threshold_db=float(args.snr_threshold_db),
    )

    receiver_lat = float(config.get("station", {}).get("receiver_lat", 0.0))
    receiver_lon = float(config.get("station", {}).get("receiver_lon", 0.0))
    tx_distances = {
        tx["id"]: haversine_km(
            receiver_lat, receiver_lon,
            float(tx["tx_lat_deg"]), float(tx["tx_lon_deg"]),
        )
        for tx in txs
    }
    tdma_offsets = match_peaks_to_txs(
        peaks, tx_distances,
        sample_rate_hz=sample_rate_hz,
        n_per_sweep=n_per_sweep,
    )

    if args.json:
        out = {
            "radiod_id": block.get("id", ""),
            "channel_name": block.get("channel_name", ""),
            "sample_rate_hz": int(sample_rate_hz),
            "sweep_rate_hz_per_s": sweep_rate_hz_per_s,
            "sweep_repetition_hz": sweep_repetition_hz,
            "n_per_sweep": n_per_sweep,
            "capture_seconds": int(args.seconds),
            "peaks": [
                {
                    "offset_samples": p.offset_samples,
                    "snr_db": round(p.snr_db, 2),
                    "correlation_power": p.correlation_power,
                }
                for p in peaks
            ],
            "tx_assignments": [
                {
                    "id": tx_id,
                    "ground_distance_km": round(tx_distances[tx_id], 2),
                    "tdma_offset_samples": tdma_offsets[tx_id],
                }
                for tx_id in tdma_offsets
            ],
        }
        print(json.dumps(out, indent=2))
        return

    print(
        f"TDMA scan on radiod={block.get('id','?')} "
        f"channel={block.get('channel_name','?')} "
        f"κ={sweep_rate_hz_per_s:.1f} Hz/s SRF={sweep_repetition_hz:.1f} Hz"
    )
    print(
        f"capture: {args.seconds} s = {len(rx_samples)} samples at "
        f"{int(sample_rate_hz)} Hz"
    )
    print(
        f"discovered {len(peaks)} peak(s) "
        f"(snr_threshold={args.snr_threshold_db:g} dB):"
    )
    for i, p in enumerate(peaks, start=1):
        print(
            f"  peak {i}: offset_samples={p.offset_samples:>7d}  "
            f"snr={p.snr_db:.1f} dB"
        )
    print("per-TX TDMA offsets (after subtracting ground delay):")
    for tx_id, offset in tdma_offsets.items():
        D = tx_distances[tx_id]
        if offset is None:
            print(f"  {tx_id}: D={D:.0f} km — no peak matched")
        else:
            print(
                f"  {tx_id}: D={D:.0f} km → tdma_offset_samples={offset}"
            )


def _handle_config(args):
    from codar_sounder import configurator
    sub = getattr(args, "config_command", None)
    if sub == "init":
        sys.exit(configurator.cmd_config_init(args))
    if sub == "edit":
        sys.exit(configurator.cmd_config_edit(args))
    print("usage: codar-sounder config {init|edit} [--non-interactive]")
    sys.exit(2)


if __name__ == "__main__":
    main()

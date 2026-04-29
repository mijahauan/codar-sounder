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

"""`codar-sounder config init|edit` — first-run wizard + edit flow (CONTRACT §14).

The interactive path is a whiptail **transmitter picker** (v0.2):
``scripts/config-wizard.sh`` lets the operator multi-select which CODAR
sites to record from the distance-ranked inventory
(``codar-sounder stations``), grouped by band into ``[[radiod]]`` blocks
and written back via ``config apply``.  Dispatched through
``sigmond.wizard_dispatch`` (same gate/contract the other recorders use).

The non-interactive path (no TTY, ``--non-interactive``, or whiptail
absent) falls back to copying the template into place and surfacing the
env-bag (``STATION_*`` / ``SIGMOND_RADIOD_STATUS``) defaults for the
operator to finish by hand.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from codar_sounder.config import DEFAULT_CONFIG_PATH

_REPO = Path(__file__).resolve().parent.parent.parent
_TEMPLATE = _REPO / "config" / "codar-sounder-config.toml.template"
_WIZARD_PATH = _REPO / "scripts" / "config-wizard.sh"


def cmd_config_init(args) -> int:
    """Dispatch to the whiptail picker when available; else copy template."""
    if not getattr(args, "non_interactive", False) and _wizard_available(args):
        # Ensure a file exists for the wizard's `config show` prefill /
        # `config apply` merge target.  Idempotent; respects --reconfig.
        target = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
        if not target.exists() or bool(getattr(args, "reconfig", False)):
            rv = _legacy_config_init(args)
            if rv != 0:
                return rv
        return _exec_wizard(args, "init")
    return _legacy_config_init(args)


def _legacy_config_init(args) -> int:
    target = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    reconfig = bool(getattr(args, "reconfig", False))

    if target.exists() and not reconfig:
        print(f"codar-sounder: config exists at {target}; use --reconfig to overwrite")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    if not _TEMPLATE.exists():
        print(f"codar-sounder: template missing: {_TEMPLATE}")
        return 1

    shutil.copy(str(_TEMPLATE), str(target))
    print(f"codar-sounder: wrote {target}")
    print("Edit the file to set:")
    print("  [station] callsign / grid_square / receiver_lat / receiver_lon")
    print("  [[radiod]] status / channel_name")
    print("    (`status` is the mDNS multicast name of the radiod —")
    print("     RADIOD-IDENTIFICATION.md §3.1)")
    print("  [[radiod.transmitter]] for each CODAR station you want to monitor")
    print("Then enable a service instance:  sudo systemctl enable codar-sounder@<reporter-id>")

    # RADIOD-IDENTIFICATION.md §4 — surface discoverable radiods so
    # the operator can paste a name straight into [[radiod]] status
    # without having to remember/look up the multicast hostname.
    discovered = _discover_radiods()
    if discovered:
        print("")
        print("Radiods broadcasting on the LAN (paste one into "
              "[[radiod]] status):")
        for svc in discovered:
            print(f"  status       = \"{svc['hostname']}\"   "
                  f"# advertised: {svc['name']!r}")
    else:
        print("")
        print("\033[33m⚠\033[0m  No radiod instances broadcasting on the "
              "local network.  Install + start radiod before the daemon")
        print("   can connect:  sudo smd install ka9q-radio")

    # Surface contract §14.3 env-bag values when present (operator-friendly).
    call = os.environ.get("STATION_CALL")
    grid = os.environ.get("STATION_GRID")
    if call or grid:
        print("")
        print("Sigmond env-bag values you can paste into [station]:")
        if call: print(f"  callsign = \"{call}\"")
        if grid: print(f"  grid_square = \"{grid}\"")
    return 0


def _discover_radiods(timeout: float = 5.0) -> list[dict]:
    """Return discovered radiods or [] on failure.

    Per RADIOD-IDENTIFICATION.md §4 — used to surface the LAN's
    available radiods to the operator during `config init`.  Each
    entry is {"name", "hostname", "address", "port"}; `hostname` is
    the mDNS multicast name (the canonical identifier).
    """
    try:
        from ka9q.discovery import discover_radiod_services
        return discover_radiod_services(timeout=timeout) or []
    except Exception:
        return []


def cmd_config_edit(args) -> int:
    target = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    if not target.exists():
        print(f"codar-sounder: no config at {target}; run `codar-sounder config init` first")
        return 1

    if getattr(args, "non_interactive", False):
        print(target.read_text())
        return 0

    # Interactive edit re-runs the transmitter picker (it prefills from the
    # existing config via `config show`), so an operator can add/drop sites
    # without hand-editing.  Falls back to $EDITOR when whiptail is absent.
    if _wizard_available(args):
        return _exec_wizard(args, "edit")

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    import subprocess
    return subprocess.run([editor, str(target)], check=False).returncode


# ---------------------------------------------------------------------------
# Whiptail wizard dispatch — delegates to sigmond.wizard_dispatch when
# importable (the canonical shared lib the recorders use), with a local
# fallback so codar-sounder still works standalone.  Mirrors
# psk-recorder/src/psk_recorder/configurator.py.
# ---------------------------------------------------------------------------

try:
    import sigmond.wizard_dispatch as _sigmond_wd
    assert _sigmond_wd.SIGMOND_WIZARD_DISPATCH_API == "1", (
        f"sigmond.wizard_dispatch API "
        f"{_sigmond_wd.SIGMOND_WIZARD_DISPATCH_API!r} != '1' "
        f"(expected by codar-sounder)"
    )
    _SIGMOND_WIZARD_LIB_SH: Optional[Path] = (
        Path(_sigmond_wd.__file__).resolve().parent / "wizard_dispatch.sh"
    )
    if not _SIGMOND_WIZARD_LIB_SH.is_file():
        _SIGMOND_WIZARD_LIB_SH = None
except (ImportError, AssertionError):
    _sigmond_wd = None
    _SIGMOND_WIZARD_LIB_SH = None


def _wizard_available(args=None) -> bool:
    """True iff the shell picker should run for this invocation.

    Defers to sigmond.wizard_dispatch.is_wizard_available when sigmond is
    importable (shared gate: --non-interactive off, stdin+stdout TTYs,
    whiptail on PATH, wizard script present + executable); falls back to
    the same checks locally when sigmond isn't installed."""
    if _sigmond_wd is not None:
        if args is None:
            import argparse as _argparse
            args = _argparse.Namespace(non_interactive=False)
        return _sigmond_wd.is_wizard_available(args, _WIZARD_PATH)

    import shutil as _shutil
    if getattr(args, "non_interactive", False):
        return False
    if not _WIZARD_PATH.is_file() or not os.access(_WIZARD_PATH, os.X_OK):
        return False
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        return False
    return _shutil.which("whiptail") is not None


def _exec_wizard(args, mode: str) -> int:
    """Hand off to scripts/config-wizard.sh, preserving --config."""
    extra_env: dict = {
        # The wizard shells back to `codar-sounder stations|config apply`;
        # use the same binary the caller invoked so a non-default --config
        # (and the editable-vs-/usr/local install split) keeps working.
        "CODAR_SOUNDER_CLI": sys.argv[0],
    }
    extra_args = [mode]
    config_arg = getattr(args, "config", None)
    if config_arg:
        extra_args += ["--config", str(config_arg)]

    if _sigmond_wd is not None:
        if _SIGMOND_WIZARD_LIB_SH is not None:
            extra_env["SIGMOND_WIZARD_LIB_SH"] = str(_SIGMOND_WIZARD_LIB_SH)
        # parse=None → the wizard pipes JSON into `config apply` itself and
        # renders its own UI; default interactive=True inherits the TTY.
        result = _sigmond_wd.exec_wizard(
            _WIZARD_PATH, extra_env=extra_env, parse=None, extra_args=extra_args,
        )
        if result.error:
            print(f"codar-sounder: wizard exec failed: {result.error}",
                  file=sys.stderr)
            return 1
        return result.returncode

    # Local fallback (sigmond not importable).
    import subprocess
    env = os.environ.copy()
    env.update(extra_env)
    try:
        return subprocess.call([str(_WIZARD_PATH)] + extra_args, env=env)
    except FileNotFoundError as exc:
        print(f"codar-sounder: wizard exec failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# CLIENT-CONTRACT §14 — JSON config-roundtrip surface.
#
# `codar-sounder config show --json [--defaults]`   reads the TOML file
#   on disk and emits it as JSON on stdout.  `--defaults` is accepted
#   but currently a no-op — codar-sounder doesn't carry a canonical
#   DEFAULTS dict; the on-disk file IS the source of truth.
#
# `codar-sounder config apply --json -`   reads a JSON dict from stdin,
#   deep-merges it into the existing TOML file, and atomically rewrites
#   the file.  Section whitelist + structural type checks only.
#
# Pattern lifted from wspr-recorder commit ad8f637 (the simpler of the
# two prior Phase 2 implementations — codar-sounder lacks a DEFAULTS
# dict, same as wspr).  Schema whitelist matches codar-sounder's
# actual sections: [station], [paths], [processing], [[radiod]],
# plus [instance] from the per-reporter migration.
#
# The `[[radiod.transmitter]]` picker that this follow-up envisioned now
# exists: `core/stations.py` is the canonical inventory (id / freq /
# sweep params / location), surfaced as `codar-sounder stations --json`;
# `scripts/config-wizard.sh` renders it as a whiptail multi-select and
# writes the chosen sites back through `config apply` (below).  That apply
# path must therefore serialize nested `[[radiod.transmitter]]`
# arrays-of-tables to valid TOML — see _serialize_toml.
# ---------------------------------------------------------------------------

import copy
import json
import sys
import tempfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from .config import DEFAULT_CONFIG_PATH


_APPLY_ALLOWED_SECTIONS = {
    "instance", "station", "paths", "processing", "radiod",
}


def cmd_config_show(args) -> int:
    """Emit the on-disk TOML as JSON on stdout.

    `--defaults` is accepted for forward-compat but doesn't merge in a
    canonical defaults dict — codar-sounder doesn't carry one (the
    live file is the source of truth).  Sigmond's wizard tolerates
    this: keys not in the file simply don't appear in the form, which
    is the expected behavior for the edit-existing flow.
    """
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        out: dict = {}
    else:
        try:
            with open(config_path, "rb") as f:
                out = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"config show: cannot read {config_path}: {exc}",
                  file=sys.stderr)
            return 2
    json.dump(out, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_config_apply(args) -> int:
    """Read a JSON dict on stdin, validate, atomically write the TOML.

    Section whitelist + structural type checks (each section must be
    a table, except `radiod` which is a list of tables).  No per-key
    type enforcement — sigmond's wizard owns input typing on its end.
    """
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"config apply: stdin is not valid JSON: {exc}",
              file=sys.stderr)
        return 2

    if not isinstance(payload, dict):
        print(f"config apply: top-level JSON must be an object, "
              f"got {type(payload).__name__}", file=sys.stderr)
        return 2

    unknown = set(payload.keys()) - _APPLY_ALLOWED_SECTIONS
    if unknown:
        print(f"config apply: section(s) not writable via apply: "
              f"{sorted(unknown)} "
              f"(allowed: {sorted(_APPLY_ALLOWED_SECTIONS)})",
              file=sys.stderr)
        return 2

    for section, fields in payload.items():
        if section == "radiod":
            if not isinstance(fields, list):
                print(f"config apply: [[radiod]] must be a list, "
                      f"got {type(fields).__name__}", file=sys.stderr)
                return 2
            continue
        if not isinstance(fields, dict):
            print(f"config apply: [{section}] must be a table, "
                  f"got {type(fields).__name__}", file=sys.stderr)
            return 2

    if config_path.is_file():
        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
    else:
        existing = {}
    merged = _deep_merge(existing, payload)

    text = _serialize_toml(merged)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.chmod(0o644)
    except PermissionError:
        pass
    tmp.replace(config_path)
    print(f"wrote {config_path}")
    return 0


# ---------------------------------------------------------------------------
# Helpers — identical to the wspr-recorder / hfdl-recorder versions.
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported TOML scalar type: {type(v).__name__}")


def _toml_inline_array(arr: list) -> str:
    parts = []
    for x in arr:
        if isinstance(x, (str, bool, int, float)):
            parts.append(_toml_scalar(x))
        else:
            parts.append(json.dumps(x))
    return "[" + ", ".join(parts) + "]"


def _serialize_toml(d: dict, parent: str = "") -> str:
    """Serialize ``d`` to a deterministic TOML string.

    Handles scalars, nested dicts (`[section.child]`), and arrays-of-
    tables (`[[section]]`).  Arrays of scalars render inline.  Keys
    sorted within each section for determinism.  Comments NOT preserved.
    """
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    nested: list[tuple[str, dict]] = []
    array_of_tables: list[tuple[str, list]] = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            nested.append((k, v))
        elif (isinstance(v, list) and v
              and all(isinstance(item, dict) for item in v)):
            array_of_tables.append((k, v))
        else:
            scalars.append((k, v))
    if scalars:
        if parent:
            lines.append(f"[{parent}]")
        for k, v in scalars:
            if isinstance(v, list):
                lines.append(f"{k} = {_toml_inline_array(v)}")
            else:
                lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    for k, sub in nested:
        header = f"{parent}.{k}" if parent else k
        lines.append(_serialize_toml(sub, parent=header))
    for k, blocks in array_of_tables:
        header = f"{parent}.{k}" if parent else k
        for block in blocks:
            lines.append(f"[[{header}]]")
            # Emit this table's scalars / inline arrays BEFORE any
            # sub-tables, so a following ``[[header.child]]`` header
            # doesn't swallow scalars that belong to this block.  A
            # nested list-of-dicts (e.g. [[radiod.transmitter]]) recurses
            # as its own array-of-tables rather than collapsing to an
            # inline array (which would emit invalid TOML).
            sub_scalars: list[str] = []
            sub_tables: list[tuple[str, object]] = []
            for bk in sorted(block.keys()):
                bv = block[bk]
                if isinstance(bv, dict):
                    sub_tables.append((bk, bv))
                elif (isinstance(bv, list) and bv
                      and all(isinstance(item, dict) for item in bv)):
                    sub_tables.append((bk, bv))
                elif isinstance(bv, list):
                    sub_scalars.append(f"{bk} = {_toml_inline_array(bv)}")
                else:
                    sub_scalars.append(f"{bk} = {_toml_scalar(bv)}")
            lines.extend(sub_scalars)
            for bk, bv in sub_tables:
                lines.append(_serialize_toml({bk: bv}, parent=header))
            lines.append("")
    return "\n".join(lines)

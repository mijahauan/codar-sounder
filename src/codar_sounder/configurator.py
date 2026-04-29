"""`codar-sounder config init|edit` — first-run wizard + edit flow (CONTRACT §14).

v0.1: minimal — copy the template into place, populate STATION_* and
SIGMOND_RADIOD_STATUS env-bag defaults if available, and tell the operator
to finish editing manually.  An interactive station picker driven by
``data/codar-stations.toml`` lands in v0.2.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from codar_sounder.config import DEFAULT_CONFIG_PATH

_REPO = Path(__file__).resolve().parent.parent.parent
_TEMPLATE = _REPO / "config" / "codar-sounder-config.toml.template"


def cmd_config_init(args) -> int:
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
    print("  [[radiod]] id / status_dns / channel_name")
    print("  [[radiod.transmitter]] for each CODAR station you want to monitor")
    print("Then enable a service instance:  sudo systemctl enable codar-sounder@<radiod-id>")

    # Surface contract §14.3 env-bag values when present (operator-friendly).
    call = os.environ.get("STATION_CALL")
    grid = os.environ.get("STATION_GRID")
    if call or grid:
        print("Sigmond env-bag values you can paste into [station]:")
        if call: print(f"  callsign = \"{call}\"")
        if grid: print(f"  grid_square = \"{grid}\"")
    return 0


def cmd_config_edit(args) -> int:
    target = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    if not target.exists():
        print(f"codar-sounder: no config at {target}; run `codar-sounder config init` first")
        return 1

    if getattr(args, "non_interactive", False):
        print(target.read_text())
        return 0

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    import subprocess
    return subprocess.run([editor, str(target)], check=False).returncode

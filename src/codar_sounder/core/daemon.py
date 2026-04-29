"""Sounder daemon — v0.1 stub.

v0.1 is a sigmond-integration scaffold.  The daemon validates config,
logs which transmitters it would dechirp, and idles in a heartbeat loop
honouring the systemd Watchdog.  The actual FMCW dechirping engine
(Kaeppler 2022 §2.1, modules core.dechirp / core.trace / core.invert)
lands in v0.2.

Notification lifecycle (matches psk-recorder@.service Type=notify):
    READY=1 once config is validated and we're idling
    STATUS=...  brief operator-visible message
    WATCHDOG=1  every loop iteration to keep systemd happy
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Optional

from codar_sounder.config import transmitters

log = logging.getLogger(__name__)


def _sd_notify(message: str) -> None:
    """Minimal sd_notify implementation — no python-systemd dependency."""
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(sock_path)
            s.sendall(message.encode("utf-8"))
    except OSError as exc:
        log.warning("sd_notify failed (%s): %s", message, exc)


class SounderDaemon:
    """v0.1 stub.  Subscribes to nothing yet; logs what it would do."""

    def __init__(self, config: dict, radiod_block: dict):
        self.config = config
        self.radiod = radiod_block
        self.transmitters = transmitters(radiod_block)
        self.coherent_seconds = int(
            config.get("processing", {}).get("coherent_seconds", 60)
        )
        self._running = False

    def run(self) -> None:
        rid = self.radiod.get("id", "default")
        chan = self.radiod.get("channel_name", "<unset>")
        tx_ids = [t.get("id", "<unnamed>") for t in self.transmitters]

        log.info("radiod=%s channel=%s transmitters=%s coherent_seconds=%d",
                 rid, chan, tx_ids, self.coherent_seconds)
        log.warning(
            "v0.1 stub: FMCW dechirping not yet implemented — "
            "this daemon idles to let you exercise the sigmond contract surface "
            "(inventory/validate/config) and the [[radiod.fragment]] apply path"
        )

        _sd_notify(f"READY=1\nSTATUS=v0.1 stub idling for radiod {rid}")

        self._running = True
        while self._running:
            for tx in self.transmitters:
                log.debug("would dechirp tx=%s freq=%d Hz",
                          tx.get("id"), tx.get("center_freq_hz"))
            _sd_notify("WATCHDOG=1")
            time.sleep(self.coherent_seconds)

    def stop(self) -> None:
        self._running = False

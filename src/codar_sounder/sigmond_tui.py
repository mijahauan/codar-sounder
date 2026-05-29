"""sigmond Receiver Channels TUI parser for codar-sounder.

Loaded by sigmond at TUI time via ``[client_features.receiver_channels]``
in ``deploy.toml``.  Pure function over a parsed config dict; no
imports from codar_sounder internals.
"""

from __future__ import annotations

from typing import Optional

from sigmond.ka9q_encoding import ENCODING_INTS, encoding_to_int


def parse_receiver_channels(
    cfg: dict,
) -> tuple[str, set[int], Optional[int]]:
    """Return ``(status_dns, configured_freqs_hz, encoding_int)`` from
    a codar-sounder per-instance config.

    codar-sounder lays out one or more [[radiod]] blocks, each with a
    [[transmitter]] sub-array carrying ``center_freq_hz`` per CODAR
    site.  The dechirper consumes complex F32 IQ
    (codar_sounder.core.stream); encoding=4 is hardcoded there with
    no operator-facing override, mirrored here as the default.
    """
    blocks = cfg.get("radiod") or []
    if isinstance(blocks, dict):
        blocks = [blocks]
    status = ""
    freqs: set[int] = set()
    encoding: Optional[int] = None
    for b in blocks:
        if not status:
            status = str(b.get("status") or "")
        for tx in (b.get("transmitter") or []):
            hz = tx.get("center_freq_hz")
            if hz is None:
                continue
            try:
                freqs.add(int(hz))
            except (TypeError, ValueError):
                continue
        if encoding is None and b.get("encoding"):
            encoding = encoding_to_int(b["encoding"])
    if encoding is None:
        encoding = ENCODING_INTS["f32"]
    return status, freqs, encoding

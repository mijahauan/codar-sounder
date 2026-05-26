"""In-place TOML config writer for ``tdma-scan --write-config``.

Targeted line-based scan that preserves comments and formatting outside
the lines it touches.  We deliberately avoid a full parse-rewrite cycle
(via tomli-w / tomlkit) because operator-edited config files are full
of comments, anchors, and whitespace conventions that those round-trip
libraries either drop or normalise.

The format we operate on is the canonical codar-sounder config:

    [[radiod]]
    id = "<radiod_id>"
    ...
        [[radiod.transmitter]]      # may be at any indentation
        id = "DUCK"
        center_freq_hz = 4537180
        ...
        tdma_offset_samples = 0     # may already exist; we replace it

For each ``(tx_id, offset)`` pair we:

  * scope to the ``[[radiod]]`` block whose ``id`` matches the requested
    ``radiod_id``;
  * find the ``[[radiod.transmitter]]`` block within that scope whose
    ``id = "<tx_id>"``;
  * if a ``tdma_offset_samples = <N>`` line already exists in that
    block, replace its value;
  * else insert a new line right after the ``id = "..."`` line.

The write is atomic (write-to-tempfile, rename) so a crash mid-write
can't truncate the operator's config.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Tuple


# Match the start of a top-level table-array entry.  We need to detect
# any new [[xxx]] header so we can leave the current radiod scope when
# a sibling array starts.
_TABLE_ARRAY_RE = re.compile(r'^\s*\[\[([^\]]+)\]\]\s*(?:#.*)?$')

# Match a `key = "value"` line where key=='id' or key=='status'.
# Tolerates surrounding whitespace and trailing comments.  Phase 6
# cutover (RADIOD-IDENTIFICATION.md §3.1) means the `[[radiod]]`
# block uses `status` instead of `id`; transmitter blocks still use
# `id` (a transmitter identifier like "LISL", not a radiod identifier).
_ID_LINE_RE = re.compile(r'^(\s*)(id|status)\s*=\s*"([^"]*)"\s*(?:#.*)?$')

# Match an existing `tdma_offset_samples = N` line so we can replace
# the integer in-place without disturbing any leading whitespace.
_TDMA_LINE_RE = re.compile(
    r'^(\s*)tdma_offset_samples\s*=\s*[-]?\d+\s*(?:#.*)?$'
)


def update_tdma_offsets_in_toml(
    config_path: Path,
    radiod_id: str,
    offsets: Mapping[str, int],
) -> Tuple[int, int]:
    """Persist discovered offsets in-place.  Atomic write.

    Returns ``(n_changed, n_inserted)`` — the count of existing
    ``tdma_offset_samples`` lines whose value was replaced, and the
    count of new lines inserted into transmitter blocks that lacked one.

    Raises:
        FileNotFoundError: if ``config_path`` does not exist.
        ValueError: if no ``[[radiod]]`` block with the requested
            ``radiod_id`` is found, or no transmitter block matches
            any of the requested ``offsets`` keys.  In either case the
            file is left untouched.
    """
    text = Path(config_path).read_text()
    new_lines, n_changed, n_inserted = _rewrite(text, radiod_id, offsets)
    if n_changed == 0 and n_inserted == 0:
        raise ValueError(
            f"no [[radiod.transmitter]] blocks within radiod_id={radiod_id!r} "
            f"matched the requested offsets ({sorted(offsets.keys())}); "
            f"nothing written"
        )
    _atomic_write(Path(config_path), "\n".join(new_lines))
    return n_changed, n_inserted


def _rewrite(
    text: str, radiod_id: str, offsets: Mapping[str, int],
) -> Tuple[list[str], int, int]:
    """Pure-text rewrite.  Exposed for unit testing without filesystem I/O."""
    lines = text.split("\n")
    out: list[str] = []

    # Two-state scope tracker:
    in_target_radiod = False     # inside [[radiod]] with id == radiod_id?
    saw_target_radiod = False
    current_tx_id: str | None = None
    current_tx_indent = ""
    pending_tdma_for_tx: str | None = None   # need to insert after id= line

    n_changed = 0
    n_inserted = 0

    def _flush_pending_insert(line_idx_into_out: int) -> int:
        """If we have a pending insert for the current TX, do it.  Returns
        new index advance (0 or 1)."""
        return 0      # actually inserted inline; this is a doc placeholder

    i = 0
    while i < len(lines):
        line = lines[i]
        m_table = _TABLE_ARRAY_RE.match(line)

        if m_table:
            # Switching scopes.  Reset per-block transmitter state.
            section = m_table.group(1).strip()
            if section == "radiod":
                in_target_radiod = False     # will be set once we see the id
                current_tx_id = None
            elif section == "radiod.transmitter":
                # New TX block under whichever radiod is current.
                current_tx_id = None
                current_tx_indent = ""
                if not in_target_radiod:
                    pass          # we still walk past, just don't act
            else:
                # Some other table — leave both scopes.
                in_target_radiod = False
                current_tx_id = None
            out.append(line)
            i += 1
            continue

        m_id = _ID_LINE_RE.match(line)
        if m_id:
            indent, key, id_value = m_id.group(1), m_id.group(2), m_id.group(3)
            # Determine whether this key belongs to a [[radiod]] or a
            # [[radiod.transmitter]] by looking back at the most recent
            # table-array header in `out`.  Phase 6: `[[radiod]]`
            # identifies itself with `status =`; the transmitter block
            # still uses `id =`.
            last_section = _last_table_array(out)
            if last_section == "radiod" and key == "status":
                if id_value == radiod_id:
                    in_target_radiod = True
                    saw_target_radiod = True
                else:
                    in_target_radiod = False
                out.append(line)
                i += 1
                continue
            if last_section == "radiod.transmitter" and key == "id":
                current_tx_id = id_value
                current_tx_indent = indent
                out.append(line)
                # If we have a discovered offset for this TX *and* there's
                # no existing tdma_offset_samples line later in this
                # block, we'll insert one when the block ends.  Mark
                # pending; the existing-line scan below clears it.
                if (
                    in_target_radiod
                    and id_value in offsets
                ):
                    # Look ahead to detect existing tdma line in this block.
                    existing_idx = _find_existing_tdma(lines, i + 1)
                    if existing_idx is None:
                        # Insert immediately after this id= line, indented
                        # to match the id line.
                        new_line = (
                            f"{indent}tdma_offset_samples = "
                            f"{offsets[id_value]}"
                        )
                        out.append(new_line)
                        n_inserted += 1
                i += 1
                continue
            # id = ... in some other context; leave alone.
            out.append(line)
            i += 1
            continue

        m_tdma = _TDMA_LINE_RE.match(line)
        if m_tdma and in_target_radiod and current_tx_id in offsets:
            indent = m_tdma.group(1)
            new_line = (
                f"{indent}tdma_offset_samples = "
                f"{offsets[current_tx_id]}"
            )
            out.append(new_line)
            n_changed += 1
            i += 1
            continue

        out.append(line)
        i += 1

    if not saw_target_radiod:
        raise ValueError(
            f"no [[radiod]] block with status={radiod_id!r} found in config"
        )
    return out, n_changed, n_inserted


def _last_table_array(lines: list[str]) -> str | None:
    """Walk `lines` backwards and return the most recent [[xxx]] table
    header (the substring inside the brackets), or None if none yet."""
    for line in reversed(lines):
        m = _TABLE_ARRAY_RE.match(line)
        if m:
            return m.group(1).strip()
    return None


def _find_existing_tdma(lines: list[str], start: int) -> int | None:
    """Look forward from `start` for an existing tdma_offset_samples line
    in the *same* [[radiod.transmitter]] block.  Return its index, or
    None if the block ends (next [[ header) before we find one."""
    for j in range(start, len(lines)):
        line = lines[j]
        if _TABLE_ARRAY_RE.match(line):
            return None
        if _TDMA_LINE_RE.match(line):
            return j
    return None


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

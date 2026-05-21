#!/usr/bin/env python3
"""F2_extreme classification diagnostic — test the multi-hop hypothesis.

The Kp-correlation analysis (2026-05-21) showed F2_extreme rate at
~35% across all geomagnetic conditions, completely uncorrelated with
Kp.  Hypothesis: ``invert()`` is misclassifying multi-hop sky-wave
returns as 1-hop with implausibly high h'.  At SEAB / 1416 km ground
distance:

    h'_apparent(N) = (1/(2N)) · sqrt(P² - D²)

A record at observed group_range = 2064 km could be:
    1-hop at h' = 751 km   (classified F2_extreme; rare conditions)
    2-hop at h' = 375 km   (normal F2; climatologically common)
    3-hop at h' = 250 km   (normal F2; climatologically common)

This script:
  1. Loads all JSONL records over a date range.
  2. Filters to mode_layer == 'F2_extreme'.
  3. For each record, computes h'_apparent under N = 1, 2, 3, 4 hop
     hypotheses.
  4. Histograms group_range distribution and per-N apparent-h' values.
  5. Reports the climatological fit (% of F2_extreme records whose
     N-hop interpretation lands in the "normal F2" band, 150-450 km).

If a majority of F2_extreme records have a plausible 2-hop or 3-hop
interpretation at normal F2 heights, the misclassification
hypothesis is supported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# "Normal F2" virtual-height band.  Davies (1990) digisonde
# conventions put F1 at 150-200 km, F2 daytime at 200-300 km,
# F2 nighttime at 300-450 km, F2_extreme above 600 km.  Anything
# in 150-450 is a climatologically common F2 reflection height.
F2_NORMAL_MIN_KM = 150.0
F2_NORMAL_MAX_KM = 450.0


def apparent_h_km(group_range_km: float, ground_distance_km: float, n_hops: int) -> float | None:
    """Compute h'_apparent under an N-hop assumption.  Returns None if
    the geometry is impossible (P < D, i.e. group_range too short for
    sky-wave at any hop count)."""
    p, d, n = group_range_km, ground_distance_km, n_hops
    if p <= d:
        return None
    discriminant = p * p - d * d
    if discriminant < 0:
        return None
    return math.sqrt(discriminant) / (2.0 * n)


def iter_records(jsonl_root: Path, start: dt.date, end: dt.date) -> Iterable[dict]:
    current = start
    while current <= end:
        path = (jsonl_root
                / f"{current.year:04d}"
                / f"{current.month:02d}"
                / f"{current.day:02d}.jsonl")
        if path.exists():
            log.info("reading %s", path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        current += dt.timedelta(days=1)


def _bin_index(value: float, bin_size: float, bin_min: float) -> int:
    return int((value - bin_min) // bin_size)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, type=lambda s: dt.date.fromisoformat(s))
    p.add_argument("--end", required=True, type=lambda s: dt.date.fromisoformat(s))
    p.add_argument(
        "--jsonl-root", type=Path,
        default=Path("/var/lib/codar-sounder/ac0g-bee1-rx888/SEAB"),
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Streaming counters.
    n_total = 0
    mode_counts: dict[str, int] = defaultdict(int)
    f2e_records: list[tuple[float, float, str]] = []   # (P, D, timestamp)
    f2e_group_range_hist: dict[int, int] = defaultdict(int)  # 100-km bins
    n_hop_apparent: dict[int, list[float]] = defaultdict(list)
    n_hop_in_normal_band: dict[int, int] = defaultdict(int)
    n_hop_in_f1_band: dict[int, int] = defaultdict(int)
    n_hop_below_F: dict[int, int] = defaultdict(int)
    n_hop_above_F2_extreme: dict[int, int] = defaultdict(int)

    for rec in iter_records(args.jsonl_root, args.start, args.end):
        n_total += 1
        mode = rec.get("mode_layer", "unknown")
        mode_counts[mode] += 1
        if mode != "F2_extreme":
            continue
        P = float(rec["group_range_km"])
        D = float(rec["ground_distance_km"])
        f2e_records.append((P, D, rec["timestamp"]))
        # 100 km bins.
        f2e_group_range_hist[_bin_index(P, 100, 0)] += 1
        for n in (1, 2, 3, 4):
            h = apparent_h_km(P, D, n)
            if h is None:
                continue
            n_hop_apparent[n].append(h)
            if F2_NORMAL_MIN_KM <= h <= F2_NORMAL_MAX_KM:
                n_hop_in_normal_band[n] += 1
            elif h < F2_NORMAL_MIN_KM:
                n_hop_below_F[n] += 1
            elif h > 600.0:
                n_hop_above_F2_extreme[n] += 1
            else:
                n_hop_in_f1_band[n] += 1   # 450-600 km, F2 nighttime / borderline

    n_f2e = len(f2e_records)
    log.info("processed %d records, %d (%.1f%%) F2_extreme",
             n_total, n_f2e, 100 * n_f2e / max(n_total, 1))

    # ─── Build report ────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# F2_extreme misclassification diagnostic\n")
    lines.append(
        f"Analysis window: {args.start} to {args.end} (UTC); "
        f"source: {args.jsonl_root}\n"
    )
    lines.append(f"Total records: {n_total:,}\n")
    lines.append("## mode_layer distribution\n")
    lines.append("| mode_layer | n | %% |")
    lines.append("|---|---|---|")
    for m in sorted(mode_counts, key=lambda k: -mode_counts[k]):
        c = mode_counts[m]
        lines.append(f"| {m} | {c:,} | {100*c/n_total:.1f} |")

    if n_f2e == 0:
        lines.append("\n## No F2_extreme records — nothing to investigate.\n")
        out = "\n".join(lines)
        if args.output:
            args.output.write_text(out)
        print(out)
        return

    # Multi-hop interpretation summary.
    lines.append("\n## Multi-hop reinterpretation of F2_extreme records\n")
    lines.append(
        f"For each F2_extreme record (n = {n_f2e}), compute "
        f"``h'_apparent = sqrt(P² - D²) / (2N)`` for N = 1, 2, 3, 4.  "
        f"Count how many records land in:\n"
    )
    lines.append(
        f"  - **normal F2 band** (h' = {F2_NORMAL_MIN_KM:.0f}–{F2_NORMAL_MAX_KM:.0f} km)\n"
        f"  - **F1 / borderline** (h' = 450–600 km)\n"
        f"  - **F2_extreme** (h' > 600 km)\n"
        f"  - **below normal F** (h' < 150 km)\n"
    )
    lines.append(
        "| N | normal F2 | F1/borderline | F2_extreme | below F | mean h' | median h' |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for n in (1, 2, 3, 4):
        norm = n_hop_in_normal_band[n]
        f1b = n_hop_in_f1_band[n]
        f2e = n_hop_above_F2_extreme[n]
        below = n_hop_below_F[n]
        vals = n_hop_apparent[n]
        if vals:
            mean_h = sum(vals) / len(vals)
            sorted_vals = sorted(vals)
            median_h = sorted_vals[len(sorted_vals) // 2]
        else:
            mean_h = 0.0
            median_h = 0.0
        lines.append(
            f"| {n} | {norm:,} ({100*norm/n_f2e:.1f}%) "
            f"| {f1b:,} ({100*f1b/n_f2e:.1f}%) "
            f"| {f2e:,} ({100*f2e/n_f2e:.1f}%) "
            f"| {below:,} ({100*below/n_f2e:.1f}%) "
            f"| {mean_h:.0f} km | {median_h:.0f} km |"
        )

    # Group-range histogram.
    lines.append("\n## F2_extreme group_range histogram (100 km bins)\n")
    lines.append("| P range (km) | count | %% |")
    lines.append("|---|---|---|")
    sorted_bins = sorted(f2e_group_range_hist.keys())
    for b in sorted_bins:
        lo = b * 100
        c = f2e_group_range_hist[b]
        lines.append(f"| {lo}–{lo+99} | {c:,} | {100*c/n_f2e:.1f} |")

    # Key climatological fit metric.
    lines.append("\n## Climatological fit\n")
    norm_2 = n_hop_in_normal_band[2]
    norm_3 = n_hop_in_normal_band[3]
    norm_either = sum(
        1 for P, D, _ in f2e_records
        if (lambda h2, h3:
            (h2 is not None and F2_NORMAL_MIN_KM <= h2 <= F2_NORMAL_MAX_KM)
            or (h3 is not None and F2_NORMAL_MIN_KM <= h3 <= F2_NORMAL_MAX_KM)
            )(apparent_h_km(P, D, 2), apparent_h_km(P, D, 3))
    )
    lines.append(
        f"- {norm_2:,} / {n_f2e:,} ({100*norm_2/n_f2e:.1f}%) F2_extreme "
        f"records have a clean **2-hop** reinterpretation in the normal F2 band "
        f"({F2_NORMAL_MIN_KM:.0f}–{F2_NORMAL_MAX_KM:.0f} km).\n"
    )
    lines.append(
        f"- {norm_3:,} / {n_f2e:,} ({100*norm_3/n_f2e:.1f}%) have a clean "
        f"**3-hop** reinterpretation in the normal F2 band.\n"
    )
    lines.append(
        f"- {norm_either:,} / {n_f2e:,} ({100*norm_either/n_f2e:.1f}%) have "
        f"**either** 2-hop OR 3-hop reinterpretation in normal F2.\n"
    )
    if norm_either / n_f2e > 0.7:
        lines.append(
            "\n**Verdict: multi-hop misclassification hypothesis is strongly "
            "supported.** A majority of records flagged as F2_extreme have a "
            "climatologically plausible multi-hop interpretation at normal "
            "F2 heights — far more likely than persistent F2_extreme "
            "conditions across 35% of CPIs."
        )
    elif norm_either / n_f2e > 0.4:
        lines.append(
            "\n**Verdict: multi-hop misclassification partially supported.** "
            "A substantial fraction of F2_extreme records fit multi-hop "
            "interpretations; the remainder may be real F2_extreme or other "
            "phenomena worth separate investigation."
        )
    else:
        lines.append(
            "\n**Verdict: multi-hop misclassification NOT supported.** "
            "Most F2_extreme records cannot be cleanly reinterpreted as "
            "multi-hop returns; the high rate has a different cause."
        )

    report = "\n".join(lines)
    if args.output:
        args.output.write_text(report)
        log.info("wrote %s", args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()

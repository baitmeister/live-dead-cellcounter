"""CSV export: per-file viability plus a per-group summary.

Export is gated on every file having been reviewed, which is the requirement
that the numbers are never produced from a blind automatic pass.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .state import ReviewState, group_sort_key, parse_name

PER_FILE_CSV = "viability.csv"
PER_GROUP_CSV = "viability_by_group.csv"


class NotReviewedError(RuntimeError):
    """Raised when export is attempted before every file has been reviewed."""


def _mean_sd_sem(values: list[float]) -> tuple[float, float | None, float | None]:
    """Mean, sample SD (n-1) and SEM. SD/SEM are None for a single replicate."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    return mean, sd, sd / math.sqrt(n)


def _fmt(value: float | None, places: int = 2) -> str:
    return "" if value is None else f"{value:.{places}f}"


def export(state: ReviewState, folder: Path | None = None) -> tuple[Path, Path]:
    """Write both CSVs. Raises NotReviewedError if any file is still pending."""
    if not state.all_reviewed():
        pending = state.unreviewed()
        raise NotReviewedError(
            f"{len(pending)} of {state.n_files} files still unreviewed "
            f"(first: {', '.join(pending[:5])}{'...' if len(pending) > 5 else ''})"
        )

    out = Path(folder) if folder else state.folder
    per_file, per_group = out / PER_FILE_CSV, out / PER_GROUP_CSV

    rows = []
    by_group: dict[str, list[float]] = defaultdict(list)

    for name in state.filenames:
        st = state[name]
        group, replicate = parse_name(name)
        total = st.total_count or 0
        alive = st.alive_count or 0
        via = 100.0 * alive / total if total > 0 else None
        if via is not None:
            by_group[group].append(via)
        rows.append(
            {
                "group": group,
                "replicate": replicate,
                "file": name,
                "total_cells": total,
                "alive_cells": alive,
                "viability_pct": _fmt(via),
            }
        )

    with per_file.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["group", "replicate", "file", "total_cells",
                        "alive_cells", "viability_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with per_group.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["group", "n_replicates", "mean_viability_pct",
             "sd_viability_pct", "sem_viability_pct"]
        )
        for group in sorted(by_group, key=group_sort_key):
            values = by_group[group]
            mean, sd, sem = _mean_sd_sem(values)
            writer.writerow([group, len(values), _fmt(mean), _fmt(sd), _fmt(sem)])

    return per_file, per_group

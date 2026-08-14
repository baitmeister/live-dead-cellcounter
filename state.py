"""Review state: per-file settings and manual edits, persisted to JSON.

Saved after every change so a review session can be interrupted and resumed.
The CSV gate lives here too: `all_reviewed()` is what the export button checks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import detect

STATE_FILENAME = "review_state.json"

_NATURAL_TOKEN = re.compile(r"(\d+)")


@dataclass
class ChannelSettings:
    """Detection settings for one channel of one file."""

    threshold: float = detect.DEFAULT_THRESHOLD
    min_signal: float = detect.DEFAULT_MIN_SIGNAL
    min_size: float = detect.DEFAULT_MIN_SIZE
    max_size: float = detect.DEFAULT_MAX_SIZE
    min_distance: int = detect.BASE_MIN_DISTANCE
    # Per-file lower black point used only in REVIEW display mode.  The upper
    # limit is deliberately fixed at the folder-wide value by the GUI.
    review_low: float | None = None
    manual_add: list[list[float]] = field(default_factory=list)
    manual_remove: list[list[float]] = field(default_factory=list)

    @property
    def n_manual_edits(self) -> int:
        return len(self.manual_add) + len(self.manual_remove)


@dataclass
class FileState:
    total: ChannelSettings = field(default_factory=ChannelSettings)
    alive: ChannelSettings = field(default_factory=ChannelSettings)
    reviewed: bool = False
    # Counts are cached on accept so export never needs to recompute images.
    total_count: int | None = None
    alive_count: int | None = None


def parse_name(filename: str) -> tuple[str, str]:
    """Split 'treatment-a.3.tif' into ('treatment-a', '3')."""
    stem = Path(filename).stem
    group, _, replicate = stem.rpartition(".")
    if not group:                      # no dot: treat whole stem as the group
        return stem, ""
    return group, replicate


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Case-insensitive natural key: sample-2 sorts before sample-10."""
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in _NATURAL_TOKEN.split(value)
        if token
    )


def group_sort_key(group: str) -> tuple[tuple[int, int | str], ...]:
    """Natural, dataset-independent group ordering."""
    return _natural_key(group)


def file_sort_key(filename: str) -> tuple:
    """Sort files naturally by parsed group, replicate, then full filename."""
    group, replicate = parse_name(filename)
    return (_natural_key(group), _natural_key(replicate), _natural_key(filename))


class ReviewState:
    """All per-file state for a folder of TIFFs, backed by a JSON file."""

    def __init__(self, folder: Path, filenames: list[str]):
        self.folder = Path(folder)
        self.filenames = sorted(filenames, key=file_sort_key)
        self.path = self.folder / STATE_FILENAME
        self.files: dict[str, FileState] = {n: FileState() for n in self.filenames}
        self.load()

    # --- persistence -----------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return                      # corrupt or unreadable: start fresh
        version = int(raw.get("version", 1))
        allowed = {item.name for item in fields(ChannelSettings)}

        def channel_settings(blob: dict) -> ChannelSettings:
            values = dict(blob)
            # Version 1 stored a free two-ended Napari contrast window.  Those
            # values may have come from per-slide auto-contrast, so deliberately
            # do not reinterpret them as the new lower-only REVIEW setting.
            if version < 2:
                values.pop("contrast", None)
            return ChannelSettings(
                **{key: value for key, value in values.items() if key in allowed}
            )

        for name, blob in raw.get("files", {}).items():
            if name not in self.files:
                continue                # file no longer present; ignore
            self.files[name] = FileState(
                total=channel_settings(blob.get("total", {})),
                alive=channel_settings(blob.get("alive", {})),
                # Version 2 introduces unique nucleus-Calcein association, so
                # cached version-1 live counts are not valid under the new
                # rule.  Preserve thresholds and manual edits but require a
                # fresh acceptance pass before export.
                reviewed=(bool(blob.get("reviewed", False))
                          if version >= 2 else False),
                total_count=(blob.get("total_count") if version >= 2 else None),
                alive_count=(blob.get("alive_count") if version >= 2 else None),
            )

    def save(self) -> None:
        blob = {
            "version": 2,
            "files": {name: asdict(st) for name, st in self.files.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(blob, indent=1))
        tmp.replace(self.path)          # atomic: never leave a half-written state

    # --- queries ---------------------------------------------------------

    def __getitem__(self, name: str) -> FileState:
        return self.files[name]

    @property
    def n_reviewed(self) -> int:
        return sum(1 for st in self.files.values() if st.reviewed)

    @property
    def n_files(self) -> int:
        return len(self.filenames)

    def all_reviewed(self) -> bool:
        """The gate for CSV export: every file must have been accepted."""
        return self.n_files > 0 and self.n_reviewed == self.n_files

    def unreviewed(self) -> list[str]:
        return [n for n in self.filenames if not self.files[n].reviewed]

    def next_unreviewed(self, after: str | None = None) -> str | None:
        """First unreviewed file at or after `after`, wrapping around."""
        pending = self.unreviewed()
        if not pending:
            return None
        if after is None:
            return pending[0]
        start = self.filenames.index(after)
        order = self.filenames[start + 1:] + self.filenames[: start + 1]
        for name in order:
            if not self.files[name].reviewed:
                return name
        return None

    # --- mutation --------------------------------------------------------

    def copy_settings_forward(self, source: str) -> int:
        """Apply the current file's slider settings to all later unreviewed files.

        Manual edits are deliberately not copied -- they are specific to one
        image and would be meaningless elsewhere.
        """
        src = self.files[source]
        start = self.filenames.index(source)
        n = 0
        for name in self.filenames[start + 1:]:
            st = self.files[name]
            if st.reviewed:
                continue
            for attr in ("total", "alive"):
                s, d = getattr(src, attr), getattr(st, attr)
                d.threshold, d.min_signal = s.threshold, s.min_signal
                d.min_size, d.max_size = s.min_size, s.max_size
                d.min_distance = s.min_distance
            n += 1
        self.save()
        return n

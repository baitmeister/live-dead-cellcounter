"""Headless verification. Run before using the GUI:

    python3 -m cellcounter.selftest

Covers what can be checked without a display: channel selection on every file,
detection sanity, the manual-edit translation logic, and the CSV export gate.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from . import detect, export
from .app import (
    CellCounter,
    _apply_pinch_zoom,
    _as_points,
    _fit_viewer,
    _limits_for_display_mode,
    _match,
    _raw_display_limits,
    _replace_points,
)
from .state import ChannelSettings, FileState, ReviewState, file_sort_key, parse_name

FAILURES: list[str] = []


def check(condition: bool, message: str) -> bool:
    if not condition:
        FAILURES.append(message)
    return condition


class _StubLayer:
    """Stands in for a napari Points layer."""

    def __init__(self):
        self.data = np.zeros((0, 2), dtype=float)
        self.size = np.zeros(0, dtype=float)
        self.selected_data = {99}
        self.slice_updates = 0
        self.refreshes = 0

    def set_view_slice(self) -> None:
        self.slice_updates += 1

    def refresh(self) -> None:
        self.refreshes += 1


def _stub_counter(file_data: detect.FileData) -> CellCounter:
    """A CellCounter with its GUI bypassed, so edit logic can be tested."""
    c = CellCounter.__new__(CellCounter)
    c.data = file_data
    c.filenames = ["stub.tif"]
    c.index = 0
    c._updating = False
    c._expected = {}
    c.points = {ch: _StubLayer() for ch in ("total", "alive")}
    c.images = {}
    c.state = ReviewState.__new__(ReviewState)
    c.state.files = {"stub.tif": FileState()}
    c.state.filenames = ["stub.tif"]
    c.state.save = lambda: None          # no state file backs this stub
    # Rendering side effects are not under test here.
    c.refresh_calls = 0

    def record_refresh() -> None:
        c.refresh_calls += 1

    c.refresh = record_refresh
    return c


def test_all_files(folder: Path) -> None:
    files = sorted((p.name for p in detect.discover_tiffs(folder)), key=file_sort_key)
    check(len(files) > 0, "no TIFF files found")
    print(f"Checking {len(files)} files\n")
    print(f"{'file':12} {'total':>6} {'alive':>6} {'viab%':>7}  notes")

    viabs = []
    t0 = time.time()
    for name in files:
        try:
            fd = detect.load_file(folder / name)
        except Exception as exc:
            FAILURES.append(f"{name}: load failed: {exc}")
            print(f"{name:12} {'LOAD FAILED':>21}  {exc}")
            continue

        h, a = fd.total.image, fd.alive.image
        check(h.shape == a.shape, f"{name}: channel shapes differ")
        check(not np.array_equal(h, a), f"{name}: two channels are identical")
        check(h.mean() > 0 and a.mean() > 0, f"{name}: a channel is blank")

        n_tot = detect.count(fd.total.candidates, detect.DEFAULT_THRESHOLD,
                             detect.DEFAULT_MIN_SIZE, detect.DEFAULT_MAX_SIZE)[1]
        alive_points = detect.count(
            fd.alive.candidates, detect.DEFAULT_THRESHOLD,
            detect.DEFAULT_MIN_SIZE, detect.DEFAULT_MAX_SIZE,
        )[0]
        total_points = detect.count(
            fd.total.candidates, detect.DEFAULT_THRESHOLD,
            detect.DEFAULT_MIN_SIZE, detect.DEFAULT_MAX_SIZE,
        )[0]
        n_ali = len(detect.associate_alive_to_nuclei(
            total_points, alive_points
        ).alive_indices)
        via = detect.viability(n_tot, n_ali)

        notes = list(fd.warnings)
        if n_tot == 0:
            notes.append("NO CELLS in Hoechst")
        if via is not None:
            viabs.append(via)
            if via > 100:
                notes.append("viability >100% - needs review")
        for ch, label in (("total", "H"), ("alive", "C")):
            frac = getattr(fd, ch).saturated_frac
            if frac > 0.001:
                notes.append(f"{label} {frac:.2%} sat")

        print(f"{name:12} {n_tot:6d} {n_ali:6d} "
              f"{'--' if via is None else f'{via:7.1f}'}  {'; '.join(notes)}")

    dt = time.time() - t0
    print(f"\nProcessed {len(files)} files in {dt:.0f}s ({dt/max(len(files),1):.1f}s each)")
    if viabs:
        print(f"Viability at default settings: min {min(viabs):.1f}%  "
              f"median {np.median(viabs):.1f}%  max {max(viabs):.1f}%")


def test_channel_identification(folder: Path) -> None:
    print("\n--- channel identification ---")
    matched = 0
    fallback = 0
    for path in sorted(detect.discover_tiffs(folder),
                       key=lambda p: file_sort_key(p.name)):
        names = detect.channel_names(path)
        recognised = (
            any(tag.casefold() in name.casefold()
                for name in names for tag in detect.HOECHST_TAGS)
            and any(tag.casefold() in name.casefold()
                    for name in names for tag in detect.CALCEIN_TAGS)
        )
        if recognised:
            matched += 1
        else:
            fallback += 1
            print(f"  {path.name}: channel aliases not found; verify page-order fallback")
    print(f"  metadata aliases matched: {matched}; page-order fallback: {fallback}")


def test_manual_edits(folder: Path) -> None:
    print("\n--- manual edit logic ---")
    name = sorted(detect.discover_tiffs(folder),
                  key=lambda p: file_sort_key(p.name))[0]
    fd = detect.load_file(name)
    c = _stub_counter(fd)
    cset = c.state.files["stub.tif"].total

    base, n_base = detect.count(fd.total.candidates, cset.threshold,
                                cset.min_size, cset.max_size)
    c._expected["total"] = base.copy()

    # Napari's pre-change event must not re-enter/refresh a half-mutated layer.
    before_adds = list(cset.manual_add)
    before_removes = list(cset.manual_remove)
    c.state.files["stub.tif"].reviewed = True
    c._on_points_edited(
        "total", SimpleNamespace(action=SimpleNamespace(name="REMOVING"))
    )
    check(cset.manual_add == before_adds and cset.manual_remove == before_removes,
          "pre-change point event must not record an edit")
    check(c.refresh_calls == 0, "pre-change point event must not refresh the layer")
    check(c.state.files["stub.tif"].reviewed,
          "pre-change point event must not invalidate acceptance")

    # Deleting two rendered points must record two removals.
    c.points["total"].data = base[2:]
    c._on_points_edited("total")
    check(len(cset.manual_remove) == 2,
          f"delete: expected 2 removals, got {len(cset.manual_remove)}")
    check(len(cset.manual_add) == 0, "delete: should not create additions")
    check(not c.state.files["stub.tif"].reviewed,
          "a completed manual edit must mark an accepted file pending")

    pts, n = detect.count(fd.total.candidates, cset.threshold, cset.min_size,
                          cset.max_size,
                          np.asarray(cset.manual_add).reshape(-1, 2),
                          np.asarray(cset.manual_remove).reshape(-1, 2))
    check(n == n_base - 2, f"delete: count {n} != {n_base - 2}")
    print(f"  delete 2 of {n_base} -> {n}")

    # Adding a point somewhere empty must record one addition.
    c._expected["total"] = pts.copy()
    c.points["total"].data = np.vstack([pts, [[5.0, 5.0]]])
    c._on_points_edited("total")
    check(len(cset.manual_add) == 1,
          f"add: expected 1 addition, got {len(cset.manual_add)}")
    pts2, n2 = detect.count(fd.total.candidates, cset.threshold, cset.min_size,
                            cset.max_size,
                            np.asarray(cset.manual_add).reshape(-1, 2),
                            np.asarray(cset.manual_remove).reshape(-1, 2))
    check(n2 == n + 1, f"add: count {n2} != {n + 1}")
    print(f"  add 1 -> {n2}")

    # Deleting that same manual point must undo the addition, not create a removal.
    n_removes = len(cset.manual_remove)
    c._expected["total"] = pts2.copy()
    c.points["total"].data = pts2[:-1] if np.allclose(pts2[-1], [5.0, 5.0]) else pts2
    c._on_points_edited("total")
    check(len(cset.manual_add) == 0,
          "undo add: manual_add should be empty again")
    check(len(cset.manual_remove) == n_removes,
          "undo add: should not have created a new removal")
    print(f"  undo that add -> adds={len(cset.manual_add)} "
          f"removes={len(cset.manual_remove)} (unchanged)")

    # Every detection control can change counts and must therefore require a
    # fresh acceptance. Display-only LUT changes are intentionally separate.
    c.sliders = {
        "total": {
            "threshold": SimpleNamespace(value=cset.threshold + 0.5),
            "min_signal": SimpleNamespace(value=cset.min_signal),
            "min_size": SimpleNamespace(value=cset.min_size),
            "max_size": SimpleNamespace(value=cset.max_size),
            "min_distance": SimpleNamespace(value=cset.min_distance + 1),
        }
    }
    c.state.files["stub.tif"].reviewed = True
    c._on_slider("total")
    check(not c.state.files["stub.tif"].reviewed,
          "a detection slider change must mark an accepted file pending")
    c.state.files["stub.tif"].reviewed = True
    c._on_spacing("total")
    check(not c.state.files["stub.tif"].reviewed,
          "a spacing change must mark an accepted file pending")

    # Edits must survive a threshold change.
    cset.threshold = 15.0
    pts3, n3 = detect.count(fd.total.candidates, cset.threshold, cset.min_size,
                            cset.max_size,
                            np.asarray(cset.manual_add).reshape(-1, 2),
                            np.asarray(cset.manual_remove).reshape(-1, 2))
    raw = detect.count(fd.total.candidates, 15.0, cset.min_size, cset.max_size)[1]
    check(n3 <= raw, "edits should not increase the count after re-thresholding")
    print(f"  after threshold 10->15: {n3} (unedited would be {raw})")


def test_size_gate(folder: Path) -> None:
    print("\n--- size gate ---")
    name = sorted(detect.discover_tiffs(folder),
                  key=lambda p: file_sort_key(p.name))[0]
    fd = detect.load_file(name)
    cands = fd.total.candidates
    wide = detect.count(cands, 10.0, 0.0, 1e6)[1]
    tight = detect.count(cands, 10.0, 20.0, 45.0)[1]
    check(tight <= wide, "tightening the size gate must not increase the count")
    check(detect.count(cands, 10.0, 1e5, 1e6)[1] == 0,
          "an impossible size gate should yield zero cells")
    d = cands.diameter
    print(f"  {name.name}: no gate {wide}, 20-45um {tight}")
    if len(d):
        print(f"  diameter um: p5 {np.percentile(d,5):.1f}  "
              f"median {np.median(d):.1f}  p95 {np.percentile(d,95):.1f}  "
              f"max {d.max():.1f}")
    else:
        print("  no permissive candidates; diameter distribution is empty")


def test_raw_signal_gate() -> None:
    print("\n--- raw signal gate ---")
    cands = detect.Candidates(
        yx=np.array([[1.0, 1.0], [2.0, 2.0]]),
        score=np.array([20.0, 20.0]),
        signal=np.array([100.0, 1000.0]),
        diameter=np.array([20.0, 20.0]),
    )
    no_gate = detect.count(cands, 10.0, 10.0, 60.0, min_signal=0.0)[1]
    default_gate = detect.count(cands, 10.0, 10.0, 60.0)[1]
    check(no_gate == 2, f"disabled raw gate should keep 2, got {no_gate}")
    check(default_gate == 1,
          f"default raw gate should reject the weak high-z peak, got {default_gate}")
    print(f"  equal z scores, raw signals 100/1000: {no_gate} -> {default_gate}")


def test_alive_association() -> None:
    print("\n--- nucleus-Calcein association ---")
    nuclei = np.array([[0.0, 0.0], [0.0, 20.0]])
    calcein = np.array([
        [0.0, 1.0],    # strongest spatial claim on nucleus 0
        [0.0, 2.0],    # duplicate peak competing for nucleus 0
        [0.0, 22.0],   # nucleus 1
        [100.0, 100.0],  # outside the radius
    ])
    match = detect.associate_alive_to_nuclei(nuclei, calcein, radius=30.0)
    check(len(match.alive_indices) == 2,
          f"expected two unique live nuclei, got {len(match.alive_indices)}")
    check(len(np.unique(match.total_indices)) == len(match.total_indices),
          "a nucleus must not receive multiple live-cell assignments")
    check(set(match.total_indices.tolist()) == {0, 1},
          f"both nuclei should be covered, got {match.total_indices}")
    check(3 in match.unmatched_alive_indices,
          "Calcein beyond 30 px must remain unmatched")
    check(np.all(match.distances <= 30.0),
          "no association may exceed the configured radius")

    none = detect.associate_alive_to_nuclei(
        np.zeros((0, 2)), calcein, radius=30.0
    )
    check(len(none.alive_indices) == 0
          and len(none.unmatched_alive_indices) == len(calcein),
          "without nuclei every Calcein point must remain unmatched")
    print("  maximum unique pairing, minimum distance, 30 px cap: OK")


def test_lut_controls() -> None:
    print("\n--- controlled display LUT ---")
    image = np.array([[0, 1000]], dtype=np.uint16)
    shared = (100.0, 1000.0)
    check(
        _limits_for_display_mode("review", shared, 300.0, image)
        == (300.0, 1000.0),
        "REVIEW must move only the lower LUT endpoint",
    )
    check(
        _limits_for_display_mode("compare", shared, 700.0, image)
        == shared,
        "COMPARE must ignore the per-file lower point",
    )
    check(
        _limits_for_display_mode("raw", shared, 700.0, image)
        == (0.0, 65535.0),
        "RAW must use the full native uint16 range",
    )
    check(
        _limits_for_display_mode("review", shared, 50.0, image)
        == (100.0, 1000.0),
        "REVIEW must not brighten below the folder lower endpoint",
    )
    check(
        _limits_for_display_mode("review", shared, 5000.0, image)
        == (999.0, 1000.0),
        "REVIEW lower endpoint must remain below the locked upper endpoint",
    )

    rng = np.random.default_rng(7)
    background = rng.normal(7900, 120, size=(512, 512)).clip(0, 65535)
    background[:32, :32] = 16000
    peak = detect.background_peak(background.astype(np.uint16))
    check(7600 <= peak <= 8200,
          f"histogram background peak should be near 7900, got {peak}")
    print(f"  REVIEW lower-only; COMPARE fixed; RAW full range; peak={peak:.0f}")


def test_export_gate(folder: Path) -> None:
    print("\n--- export gate and group maths ---")
    names = [p.name for p in detect.discover_tiffs(folder)]
    with tempfile.TemporaryDirectory() as tmp:
        st = ReviewState(Path(tmp), names)
        try:
            export.export(st)
            FAILURES.append("export gate did not fire on an unreviewed set")
        except export.NotReviewedError:
            print(f"  refused export with {st.n_files} unreviewed: OK")

        for i, n in enumerate(st.filenames):
            f = st[n]
            f.total_count, f.alive_count, f.reviewed = 100, 50 + (i % 5), True
        # One file left pending must still block the whole export.
        st[st.filenames[0]].reviewed = False
        try:
            export.export(st)
            FAILURES.append("export gate did not fire with 1 file pending")
        except export.NotReviewedError:
            print("  refused export with 1 file pending: OK")

        st[st.filenames[0]].reviewed = True
        per_file, per_group = export.export(st)
        rows = per_file.read_text().strip().splitlines()
        check(len(rows) == st.n_files + 1,
              f"expected {st.n_files + 1} CSV rows, got {len(rows)}")
        groups = per_group.read_text().strip().splitlines()
        expected_groups = len({parse_name(name)[0] for name in names})
        check(len(groups) == expected_groups + 1,
              f"expected {expected_groups} groups + header, got {len(groups)}")
        print(f"  exported {len(rows)-1} file rows and {len(groups)-1} group rows: OK")

        # Zero total cells must not raise, and must leave viability blank.
        st[st.filenames[0]].total_count = 0
        per_file, _ = export.export(st)
        first = per_file.read_text().splitlines()[1]
        check(first.endswith(","), f"zero-total row should have blank viability: {first}")
        print("  zero-cell file exports blank viability rather than crashing: OK")


def test_state_migration() -> None:
    print("\n--- review-state LUT migration ---")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        old = {
            "version": 1,
            "files": {
                "sample.1.tif": {
                    "total": {
                        "threshold": 17.0,
                        "contrast": [123.0, 456.0],
                        "manual_add": [[1.0, 2.0]],
                    },
                    "alive": {"contrast": [234.0, 567.0]},
                    "reviewed": True,
                    "total_count": 10,
                    "alive_count": 5,
                }
            },
        }
        (folder / "review_state.json").write_text(json.dumps(old))
        state = ReviewState(folder, ["sample.1.tif"])
        check(state["sample.1.tif"].total.threshold == 17.0,
              "state migration must preserve detection thresholds")
        check(state["sample.1.tif"].total.manual_add == [[1.0, 2.0]],
              "state migration must preserve manual edits")
        check(state["sample.1.tif"].total.review_low is None,
              "legacy two-ended auto-contrast must not become REVIEW low")
        check(not state["sample.1.tif"].reviewed
              and state["sample.1.tif"].alive_count is None,
              "association migration must invalidate legacy accepted counts")
        state.save()
        saved = json.loads((folder / "review_state.json").read_text())
        check(saved["version"] == 2,
              "new review state must use the lower-only LUT schema")
        check("contrast" not in saved["files"]["sample.1.tif"]["total"],
              "legacy free contrast window must be removed on save")
    print("  thresholds/edits preserved; legacy LUT/count acceptance reset: OK")


def test_helpers() -> None:
    print("\n--- helpers ---")
    a = np.array([[0.0, 0.0], [100.0, 100.0]])
    b = np.array([[0.5, 0.5]])
    m = _match(a, b)
    check(bool(m[0]) and not bool(m[1]), "_match tolerance is wrong")
    check(len(_as_points([])) == 0, "_as_points should handle an empty list")
    check(_as_points([[1.0, 2.0]]).shape == (1, 2), "_as_points shape is wrong")
    check(detect.viability(0, 0) is None, "viability of an empty image should be None")
    check(abs(detect.viability(200, 100) - 50.0) < 1e-9, "viability maths is wrong")
    check(parse_name("treatment-a.3.tiff") == ("treatment-a", "3"),
          "filename parsing is wrong")
    ordered = sorted(["sample-10.tif", "sample-2.TIF", "sample-1.tiff"],
                     key=file_sort_key)
    check(ordered == ["sample-1.tiff", "sample-2.TIF", "sample-10.tif"],
          f"natural filename sorting is wrong: {ordered}")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in ("a.tif", "b.TIF", "c.tiff", "d.TIFF", "ignore.png"):
            (folder / name).touch()
        found = sorted(path.name for path in detect.discover_tiffs(folder))
        check(found == ["a.tif", "b.TIF", "c.tiff", "d.TIFF"],
              f"TIFF suffix discovery is wrong: {found}")

    layer = _StubLayer()
    _replace_points(layer, np.array([[3.0, 4.0]]), np.array([12.0]))
    check(layer.selected_data == set(), "point replacement must clear stale selection")
    check(len(layer.data) == len(layer.size) == 1,
          "point replacement must keep data and size arrays aligned")
    check(layer.slice_updates == 1,
          "point replacement must rebuild the Napari view slice")
    check(layer.refreshes == 0,
          "point replacement must not queue a redundant asynchronous refresh")

    camera = SimpleNamespace(zoom=1.0, mouse_zoom=False)
    viewer = SimpleNamespace(camera=camera, reset_calls=[])
    viewer.reset_view = lambda **kwargs: viewer.reset_calls.append(kwargs)
    _apply_pinch_zoom(viewer, 0.1)
    check(camera.zoom > 1.0, "positive pinch delta must zoom in")
    _apply_pinch_zoom(viewer, -0.1)
    check(camera.zoom < 1.02, "negative pinch delta must zoom back out")
    _fit_viewer(viewer)
    check(camera.mouse_zoom, "fit must leave interactive zoom enabled")
    check(viewer.reset_calls == [{"margin": 0.02}],
          "fit must center the complete image with the configured margin")
    check(_raw_display_limits(np.zeros((2, 2), dtype=np.uint16)) == (0.0, 65535.0),
          "RAW display must use the full native uint16 range")
    check(_raw_display_limits(np.array([[2.0, 5.0]])) == (2.0, 5.0),
          "RAW display must use the native range for floating-point images")
    print("  helpers OK")


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"Self-test on {folder}\n" + "=" * 60)
    test_helpers()
    test_channel_identification(folder)
    test_size_gate(folder)
    test_raw_signal_gate()
    test_alive_association()
    test_lut_controls()
    test_manual_edits(folder)
    test_state_migration()
    test_export_gate(folder)
    print("\n" + "=" * 60)
    test_all_files(folder)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} problem(s)):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

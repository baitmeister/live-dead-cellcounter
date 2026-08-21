"""napari review app: live threshold/size preview, click-to-edit, CSV export.

Layout
    Hoechst             blue image layer
    Total cells overlay cyan points, marker size = measured cell diameter
    Calcein             green image layer, additive blending -> merged view
    Alive cells overlay yellow points

Every detection slider except spacing is a pure filter over precomputed
candidates, so preview is instant.  Display has three controlled modes: a
lower-only REVIEW black point, a folder-wide comparable raw window, and the full
native RAW range.  Images are only reprocessed when a new file is loaded, and
the next file is prefetched on a worker thread so navigation feels immediate.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import detect, export
from .detect import FileData
from .state import ChannelSettings, ReviewState, file_sort_key

# Manual edits are matched to rendered points within this radius (px).
MATCH_RADIUS = 6.0

# Dock sizing: wide enough that no control is clipped at the default font.
PANEL_WIDTH = 400
HIST_HEIGHT = 240
FIT_MARGIN = 0.02

CHANNELS = ("total", "alive")
CHANNEL_LABEL = {"total": "Total (Hoechst)", "alive": "Alive (Calcein)"}
LAYER_NAME = {
    "total": "Total cells overlay",
    "alive": "Alive cells overlay",
}
IMAGE_NAME = {"total": "Hoechst", "alive": "Calcein"}
POINT_COLOR = {"total": "cyan", "alive": "yellow"}
IMAGE_CMAP = {"total": "blue", "alive": "green"}

# Napari renders the last layer in its model at the top and reverses that model
# for the visible layer list.  Keep the requested user-facing order explicit so
# render-order implementation details do not leak into the UI definition.
LAYER_LIST_TOP_TO_BOTTOM = (
    IMAGE_NAME["total"],
    LAYER_NAME["total"],
    IMAGE_NAME["alive"],
    LAYER_NAME["alive"],
)

# This is also the single source of truth for the reference shown immediately
# below Napari's layer list. Delete/Backspace are handled by the active Points
# layer; every other entry is bound on the viewer in ``_bind_keys``.
SHORTCUT_REFERENCE = (
    ("q", "Hoechst visibility"),
    ("w", "Total overlay visibility"),
    ("e", "Calcein visibility"),
    ("r", "Alive overlay visibility"),
    ("1", "Hoechst view"),
    ("2", "Calcein view"),
    ("3", "Merged view"),
    ("Space", "Toggle both overlays"),
    ("g", "Toggle grid"),
    ("f", "Fit image"),
    ("n", "Accept + next"),
    ("b", "Back"),
    ("Delete / Backspace", "Remove selected marker"),
)

LUT_REVIEW = "review"
LUT_COMPARE = "compare"
LUT_RAW = "raw"
LUT_MODES = (LUT_REVIEW, LUT_COMPARE, LUT_RAW)

VIEW_HOECHST = "hoechst"
VIEW_CALCEIN = "calcein"
VIEW_MERGED = "merged"
CHANNEL_VIEWS = (VIEW_HOECHST, VIEW_CALCEIN, VIEW_MERGED)


def _as_points(data) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr.reshape(-1, 2) if arr.size else np.zeros((0, 2), dtype=float)


def _match(a: np.ndarray, b: np.ndarray, radius: float = MATCH_RADIUS) -> np.ndarray:
    """Boolean mask over `a`: True where a point has a partner in `b`."""
    if not len(a):
        return np.zeros(0, dtype=bool)
    if not len(b):
        return np.zeros(len(a), dtype=bool)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return d2.min(axis=1) <= radius**2


def _replace_points(layer, points: np.ndarray, sizes: np.ndarray) -> None:
    """Atomically replace a Napari points view without stale selections.

    Napari 0.8 can retain view/selection indices from the previous data array.
    If the new array is shorter, a later zoom redraw then indexes past its end.
    Clear selection and synchronously rebuild the view slice after both data and
    per-point sizes have been replaced.
    """
    layer.selected_data = set()
    layer.data = points
    layer.size = sizes
    layer.set_view_slice()


def _apply_pinch_zoom(viewer, delta: float) -> None:
    """Apply one macOS native-pinch increment to the Napari camera."""
    # Qt reports positive values when fingers spread (zoom in). This matches
    # Vispy's ``zoom(1 - delta)`` convention, expressed here in Napari's
    # pixels-per-world-unit camera zoom.
    denominator = max(0.05, 1.0 - float(delta))
    factor = float(np.clip(1.0 / denominator, 0.5, 2.0))
    viewer.camera.zoom = max(float(viewer.camera.zoom) * factor, 1e-6)


def _fit_viewer(viewer) -> None:
    """Center the complete image in the available canvas."""
    viewer.camera.mouse_zoom = True
    viewer.reset_view(margin=FIT_MARGIN)


def _raw_display_limits(image: np.ndarray) -> tuple[float, float]:
    """Native, unstretched intensity range for an image dtype."""
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return float(info.min), float(info.max)
    lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _limits_for_display_mode(
    mode: str,
    shared_limits: tuple[float, float],
    review_low: float,
    image: np.ndarray,
) -> tuple[float, float]:
    """Display limits for one of the three controlled viewing modes."""
    if mode == LUT_RAW:
        return _raw_display_limits(image)
    lo, hi = map(float, shared_limits)
    if mode == LUT_COMPARE:
        return lo, hi
    if mode == LUT_REVIEW:
        return float(np.clip(review_low, lo, max(lo, hi - 1.0))), hi
    raise ValueError(f"unknown display mode: {mode}")


class CellCounter:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        filenames = sorted(
            (p.name for p in detect.discover_tiffs(self.folder)), key=file_sort_key
        )
        if not filenames:
            raise SystemExit(f"No TIFF files found in {self.folder}")

        self.state = ReviewState(self.folder, filenames)
        self.filenames = self.state.filenames
        self._display_limits = detect.shared_display_limits(
            [self.folder / name for name in self.filenames]
        )
        self.index = 0
        self.data: FileData | None = None

        self._cache: dict[str, FileData] = {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._updating = False          # guards programmatic layer writes
        self._expected: dict[str, np.ndarray] = {}   # last points we rendered
        self._jump_sig: tuple = ()                   # reviewed-flags of the jump list
        self._lut_mode = LUT_REVIEW
        self._channel_view = VIEW_MERGED
        self._points_visible = True
        self._auto_review_low: dict[str, float] = {}
        self._last_association: detect.Associations | None = None

        self._build_viewer()
        self._build_controls()
        self._build_lut_controls()
        self._bind_keys()
        self._install_trackpad_zoom()

        start = self.state.next_unreviewed() or self.filenames[0]
        self.load(self.filenames.index(start))
        self._schedule_initial_fit()

    # --- construction ----------------------------------------------------

    def _build_viewer(self) -> None:
        import napari

        self.viewer = napari.Viewer(title="Cell Viability Counter")
        blank = np.zeros((4, 4), dtype=np.uint16)

        self.images = {}
        for ch in CHANNELS:
            self.images[ch] = self.viewer.add_image(
                blank,
                name=IMAGE_NAME[ch],
                colormap=IMAGE_CMAP[ch],
                blending="additive",
                opacity=1.0,
                gamma=1.0,
                interpolation2d="nearest",
            )
            # Image transforms would visually detach a channel from the other
            # image and from the point coordinates.  Navigation remains a
            # viewer/camera operation, so image layers themselves stay locked.
            self.images[ch].editable = False

        self.points = {}
        for ch in CHANNELS:
            layer = self.viewer.add_points(
                np.zeros((0, 2), dtype=float),
                name=LAYER_NAME[ch],
                border_color=POINT_COLOR[ch],   # renamed from edge_color in napari 0.8
                face_color="transparent",
                border_width=0.12,
                size=14,
                symbol="o",
                opacity=1.0,
                blending="translucent_no_depth",
            )
            layer.events.data.connect(
                lambda event, ch=ch: self._on_points_edited(ch, event)
            )
            self.points[ch] = layer

        for ch in CHANNELS:
            self.images[ch].events.contrast_limits.connect(
                lambda event, ch=ch: self._on_contrast(ch)
            )

        # Napari's generic controls include per-slide auto-contrast, gamma,
        # transforms, blending, and marker-style changes.  Those are useful in
        # a general image viewer but conflict with a controlled counting review.
        # Purpose-built channel, LUT, and point-edit controls are provided in
        # this application instead; the layer list remains for visibility.
        generic_controls = self.viewer.window._qt_viewer.dockLayerControls
        generic_controls.hide()
        generic_controls.toggleViewAction().setVisible(False)
        self.viewer.window._qt_viewer.layerButtons.hide()
        self.viewer.window._qt_viewer.viewerButtons.hide()
        self._build_shortcut_reference()
        self._order_for_overlay()

    def _build_shortcut_reference(self) -> None:
        """Place the complete app shortcut reference below the layer list."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
        )

        panel = QWidget()
        panel.setObjectName("cellCounterShortcutReference")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 7, 0, 0)
        layout.setSpacing(4)

        heading = QLabel("Keyboard shortcuts")
        heading.setStyleSheet("font-weight: bold;")
        layout.addWidget(heading)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(2)

        # Two compact columns keep the full list visible without squeezing the
        # four layer rows out of the standard left dock.
        compact_shortcuts = SHORTCUT_REFERENCE[:-1]
        rows = (len(compact_shortcuts) + 1) // 2
        for index, (key, description) in enumerate(compact_shortcuts):
            column_group, row = divmod(index, rows)
            key_label = QLabel(key)
            key_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            key_label.setStyleSheet("font-weight: bold;")
            description_label = QLabel(description)
            grid.addWidget(key_label, row, column_group * 2)
            grid.addWidget(description_label, row, column_group * 2 + 1)
            grid.setColumnStretch(column_group * 2 + 1, 1)
        layout.addLayout(grid)

        # Keep the one long key name on its own row so it does not widen and
        # squeeze either compact column above it.
        key, description = SHORTCUT_REFERENCE[-1]
        final_row = QHBoxLayout()
        final_row.setContentsMargins(0, 0, 0, 0)
        final_row.setSpacing(7)
        key_label = QLabel(key)
        key_label.setStyleSheet("font-weight: bold;")
        final_row.addWidget(key_label)
        final_row.addWidget(QLabel(description), 1)
        layout.addLayout(final_row)

        layer_list_panel = self.viewer.window._qt_viewer.dockLayerList.widget()
        layer_list_panel.layout().addWidget(panel)
        self.shortcut_reference = panel

    def _build_controls(self) -> None:
        from magicgui.widgets import (
            ComboBox, Container, FloatSlider, Label, PushButton, Slider,
        )
        from qtpy.QtWidgets import QButtonGroup

        # Separate single-line labels: one multi-line QLabel gets its first line
        # clipped by the scroll area's height calculation.
        self.w_file = Label(value="")
        self.w_progress = Label(value="")
        self.w_counts = Label(value="")
        self.w_warn = Label(value="")
        self.w_warn.native.setWordWrap(True)
        self.w_file.native.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.w_counts.native.setStyleSheet("font-size: 14px; padding: 4px 0;")
        self.w_warn.native.setStyleSheet("color: #e0a030;")

        # Jump straight to any file. Entries are marked with their review
        # status, so the dropdown doubles as a progress overview.
        self.w_jump = ComboBox(choices=self._jump_choices(), label="jump to")
        self.w_jump.native.setMaxVisibleItems(20)
        self.w_jump.changed.connect(self._on_jump)
        jump_row = Container(widgets=[self.w_jump], labels=True)
        jump_row.native.layout().setContentsMargins(0, 4, 0, 4)

        self.sliders: dict[str, dict] = {}
        blocks = [self.w_file, self.w_progress, self.w_counts, jump_row,
                  self.w_warn]
        detection_blocks = []

        # Labels are kept short so the label column does not force the dock wide.
        for ch in CHANNELS:
            thr = FloatSlider(value=detect.DEFAULT_THRESHOLD, min=3.0, max=60.0,
                              step=0.5, label="thresh z")
            raw = FloatSlider(value=detect.DEFAULT_MIN_SIGNAL, min=0.0,
                              max=20000.0, step=100.0, label="min signal")
            lo = FloatSlider(value=detect.DEFAULT_MIN_SIZE, min=0.0, max=120.0,
                             step=1.0, label="min um")
            hi = FloatSlider(value=detect.DEFAULT_MAX_SIZE, min=0.0, max=200.0,
                             step=1.0, label="max um")
            sep = Slider(value=detect.BASE_MIN_DISTANCE, min=3, max=25,
                         label="spacing")
            raw.native.setToolTip(
                "Minimum background-subtracted raw camera signal; prevents "
                "weak noise from passing only because its z-score is high."
            )
            for w in (thr, raw, lo, hi):
                w.changed.connect(lambda _=None, ch=ch: self._on_slider(ch))
            sep.changed.connect(lambda _=None, ch=ch: self._on_spacing(ch))
            self.sliders[ch] = {"threshold": thr, "min_signal": raw, "min_size": lo,
                                "max_size": hi, "min_distance": sep}

            heading = Label(value=CHANNEL_LABEL[ch])
            heading.native.setStyleSheet(
                f"font-weight: bold; color: {POINT_COLOR[ch]}; padding-top: 6px;"
            )
            group = Container(widgets=[heading, thr, raw, lo, hi, sep], labels=True)
            group.native.layout().setContentsMargins(0, 2, 0, 2)
            group.native.layout().setSpacing(2)
            detection_blocks.append(group)

        edit_heading = Label(value="Manual correction")
        edit_heading.native.setStyleSheet("font-weight: bold; padding-top: 6px;")
        self.b_pan = PushButton(text="Pan / zoom")
        self.b_add_total = PushButton(text="Add total")
        self.b_add_alive = PushButton(text="Add alive")
        self.b_select_total = PushButton(text="Select/remove total")
        self.b_select_alive = PushButton(text="Select/remove alive")

        # Manual tools are modes, not one-shot actions. Keep exactly one checked
        # and give the active tool the same conspicuous highlight as the LUT mode
        # buttons, so the next click's effect is always apparent.
        self.manual_mode_buttons = {
            "pan": self.b_pan,
            "add_total": self.b_add_total,
            "add_alive": self.b_add_alive,
            "select_total": self.b_select_total,
            "select_alive": self.b_select_alive,
        }
        self.manual_mode_group = QButtonGroup()
        self.manual_mode_group.setExclusive(True)
        manual_mode_style = (
            "QPushButton { padding: 3px 5px; margin: 0; }"
            "QPushButton:checked { background: #3478c7; color: white; "
            "border: 1px solid #79b7ff; font-weight: bold; }"
        )
        for button in self.manual_mode_buttons.values():
            button.native.setCheckable(True)
            button.native.setStyleSheet(manual_mode_style)
            self.manual_mode_group.addButton(button.native)
        self.b_pan.native.setChecked(True)

        self.b_pan.changed.connect(lambda _=None: self._activate_pan_zoom())
        self.b_add_total.changed.connect(
            lambda _=None: self._activate_point_mode("total", "add")
        )
        self.b_add_alive.changed.connect(
            lambda _=None: self._activate_point_mode("alive", "add")
        )
        self.b_select_total.changed.connect(
            lambda _=None: self._activate_point_mode("total", "select")
        )
        self.b_select_alive.changed.connect(
            lambda _=None: self._activate_point_mode("alive", "select")
        )
        self.b_select_total.native.setToolTip(
            "Select a total-cell marker, then press Delete or Backspace."
        )
        self.b_select_alive.native.setToolTip(
            "Select a live-cell marker, then press Delete or Backspace."
        )
        edit_row_1 = Container(
            widgets=[self.b_add_total, self.b_add_alive],
            layout="horizontal", labels=False,
        )
        edit_row_2 = Container(
            widgets=[self.b_select_total, self.b_select_alive],
            layout="horizontal", labels=False,
        )
        for row in (edit_row_1, edit_row_2):
            row.native.layout().setContentsMargins(0, 0, 0, 0)
            row.native.layout().setSpacing(1)
        self.manual_edit_rows = (edit_row_1, edit_row_2)
        self.manual_tools = Container(
            widgets=[self.b_pan, edit_row_1, edit_row_2], labels=False
        )
        self.manual_tools.native.layout().setContentsMargins(0, 0, 0, 0)
        self.manual_tools.native.layout().setSpacing(1)
        detection_blocks += [edit_heading, self.manual_tools]

        self.b_copy = PushButton(text="Copy settings forward")
        self.b_reset = PushButton(text="Reset this file")
        self.b_fit = PushButton(text="Fit image to window")
        self.b_prev = PushButton(text="< Back")
        # "&&" because Qt eats a single & as a mnemonic marker.
        self.b_next = PushButton(text="Accept && Next >")
        self.b_export = PushButton(text="Export CSV")
        self.b_copy.changed.connect(self._copy_forward)
        self.b_reset.changed.connect(self._reset_file)
        self.b_fit.changed.connect(lambda _=None: self._fit_view())
        self.b_prev.changed.connect(lambda _=None: self.step(-1))
        self.b_next.changed.connect(lambda _=None: self.accept_and_next())
        self.b_export.changed.connect(self._export)

        # Stacked rather than side by side: two buttons in a row need more width
        # than the dock has, which pushed "Accept & Next" off-screen entirely.
        blocks += [self.b_copy, self.b_reset, self.b_fit, self.b_prev, self.b_next,
                   self.b_export]

        # magicgui's own scrollable=True is unusable here: it returns the inner
        # widget as .native, so napari docks that and drops the scroll area.
        # The content then sits at (0, 0) of the dock -- underneath the title
        # bar, hiding the top row -- and keeps its full sizeHint width, spilling
        # past the right edge. Wrapping in a QScrollArea ourselves gives napari
        # a widget it can lay out, and setWidgetResizable makes the content
        # follow the dock width instead of overflowing it.
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QScrollArea

        def make_scrolled_panel(widgets):
            panel = Container(widgets=widgets, labels=False)
            layout = panel.native.layout()
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(4)
            scroll = QScrollArea()
            scroll.setWidget(panel.native)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            return panel, scroll

        self.panel, self.scroll = make_scrolled_panel(blocks)

        dock = self.viewer.window.add_dock_widget(
            self.scroll, area="right", name="Review"
        )
        # Wide enough that no control is clipped, narrow enough to leave the
        # canvas usable; the user can still drag the splitter.
        dock.setMinimumWidth(PANEL_WIDTH)

        self.detection_panel, self.detection_scroll = make_scrolled_panel(
            detection_blocks
        )
        detection_dock = self.viewer.window.add_dock_widget(
            self.detection_scroll, area="right", name="Detection & Manual"
        )
        detection_dock.setMinimumWidth(PANEL_WIDTH)
        self.detection_dock = detection_dock

        # The matplotlib canvas is a raw Qt widget, which a magicgui Container
        # will not accept, so it docks separately.
        self.hist = _Histogram()
        if self.hist.widget is not None:
            hist_dock = self.viewer.window.add_dock_widget(
                self.hist.widget, area="right", name="Size distribution"
            )
            hist_dock.setMinimumHeight(HIST_HEIGHT)
            self.hist.widget.setMinimumHeight(HIST_HEIGHT)

    def _build_lut_controls(self) -> None:
        """Controlled channel and lower-only LUT controls.

        This intentionally uses exclusive push-button strips instead of menus:
        the active channel view and display interpretation are always visible.
        """
        from qtpy.QtCore import Qt, QTimer
        from qtpy.QtWidgets import (
            QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSlider,
            QSpinBox, QVBoxLayout, QWidget,
        )

        panel = QWidget()
        lut_layout = QVBoxLayout(panel)
        lut_layout.setContentsMargins(10, 8, 10, 8)
        lut_layout.setSpacing(6)

        heading = QLabel("Display LUT")
        heading.setStyleSheet("font-weight: bold;")
        lut_layout.addWidget(heading)

        checked_style = (
            "QPushButton { padding: 5px 7px; }"
            "QPushButton:checked { background: #3478c7; color: white; "
            "font-weight: bold; }"
        )

        def button_strip(label: str, specs: list[tuple[str, str]]):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            layout.addWidget(QLabel(label))
            group = QButtonGroup(row)
            group.setExclusive(True)
            buttons = {}
            for key, text in specs:
                button = QPushButton(text)
                button.setCheckable(True)
                button.setStyleSheet(checked_style)
                group.addButton(button)
                layout.addWidget(button, 1)
                buttons[key] = button
            return row, group, buttons

        channel_row, self.channel_button_group, self.channel_buttons = button_strip(
            "Channel:",
            [(VIEW_HOECHST, "Hoechst"), (VIEW_CALCEIN, "Calcein"),
             (VIEW_MERGED, "Merged")],
        )
        for mode, button in self.channel_buttons.items():
            button.clicked.connect(
                lambda _checked=False, mode=mode: self._set_channel_view(mode)
            )
        self.channel_buttons[self._channel_view].setChecked(True)
        lut_layout.addWidget(channel_row)

        mode_row, self.lut_button_group, self.lut_buttons = button_strip(
            "View:",
            [(LUT_REVIEW, "Review"), (LUT_COMPARE, "Compare"),
             (LUT_RAW, "RAW")],
        )
        for mode, button in self.lut_buttons.items():
            button.clicked.connect(
                lambda _checked=False, mode=mode: self._apply_lut_mode(mode)
            )
        self.lut_buttons[self._lut_mode].setChecked(True)
        lut_layout.addWidget(mode_row)

        self.w_lut_mode = QLabel("")
        self.w_lut_mode.setWordWrap(True)
        self.w_lut_mode.setStyleSheet("color: #aaaaaa;")
        lut_layout.addWidget(self.w_lut_mode)

        self.lut_sliders = {}
        self.lut_spinboxes = {}
        for ch in CHANNELS:
            folder_lo, folder_hi = self._display_limits[ch]
            minimum = int(np.ceil(folder_lo))
            maximum = max(minimum, int(np.floor(folder_hi - 1.0)))

            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)
            layout.addWidget(QLabel("Hoechst background" if ch == "total"
                                    else "Calcein background"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setSingleStep(1)
            slider.setPageStep(max(1, (maximum - minimum) // 100))
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(1)
            spin.setFixedWidth(82)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            spin.valueChanged.connect(
                lambda value, ch=ch: self._on_lut_lower(ch, value)
            )
            tooltip = (
                "REVIEW display black point only. It can move upward from the "
                "folder value to dim background; the upper limit stays fixed. "
                "Detection and exported counts never use this value."
            )
            slider.setToolTip(tooltip)
            spin.setToolTip(tooltip)
            layout.addWidget(slider, 1)
            layout.addWidget(spin)
            self.lut_sliders[ch] = slider
            self.lut_spinboxes[ch] = spin
            lut_layout.addWidget(row)

        self.b_reset_lut = QPushButton("Auto-set background from histogram")
        self.b_reset_lut.setToolTip(
            "Set each REVIEW black point to the dominant background peak for "
            "the current slide."
        )
        self.b_reset_lut.clicked.connect(self._reset_lut)
        lut_layout.addWidget(self.b_reset_lut)

        self.lut_hist = _LUTHistogram()
        if self.lut_hist.widget is not None:
            self.lut_hist.widget.setMinimumHeight(210)
            lut_layout.addWidget(self.lut_hist.widget)
        self._lut_hist_timer = QTimer(panel)
        self._lut_hist_timer.setSingleShot(True)
        self._lut_hist_timer.setInterval(75)
        self._lut_hist_timer.timeout.connect(self._draw_lut_histogram_now)

        note = QLabel("Display only — counts always use the original TIFF pixels.")
        note.setWordWrap(True)
        note.setStyleSheet("font-style: italic; color: #aaaaaa;")
        lut_layout.addWidget(note)

        self.lut_panel = panel
        lut_dock = self.viewer.window.add_dock_widget(
            panel, area="left", name="Display LUT"
        )
        lut_dock.setMinimumWidth(PANEL_WIDTH)
        self.lut_dock = lut_dock
        self._set_lut_status()

    def _bind_keys(self) -> None:
        v = self.viewer

        # q/w are unused by Napari. e/r are context-specific defaults for
        # Labels/Shapes editing, layer types this controlled viewer never
        # creates. These viewer bindings therefore have no functional conflict
        # with this app's Image and Points layers.
        @v.bind_key("q", overwrite=True)
        def toggle_hoechst(_):
            self._toggle_layer_visibility(IMAGE_NAME["total"])

        @v.bind_key("w", overwrite=True)
        def toggle_total_overlay(_):
            self._toggle_layer_visibility(LAYER_NAME["total"])

        @v.bind_key("e", overwrite=True)
        def toggle_calcein(_):
            self._toggle_layer_visibility(IMAGE_NAME["alive"])

        @v.bind_key("r", overwrite=True)
        def toggle_alive_overlay(_):
            self._toggle_layer_visibility(LAYER_NAME["alive"])

        @v.bind_key("1", overwrite=True)
        def solo_total(_):
            self._set_channel_view(VIEW_HOECHST)

        @v.bind_key("2", overwrite=True)
        def solo_alive(_):
            self._set_channel_view(VIEW_CALCEIN)

        @v.bind_key("3", overwrite=True)
        def merge(_):
            self._set_channel_view(VIEW_MERGED)

        @v.bind_key("Space", overwrite=True)
        def toggle_points(_):
            self._points_visible = not self._points_visible
            self._set_channel_view(self._channel_view)

        @v.bind_key("g", overwrite=True)
        def grid(_):
            self._toggle_grid()

        @v.bind_key("n", overwrite=True)
        def nxt(_):
            self.accept_and_next()

        @v.bind_key("b", overwrite=True)
        def back(_):
            self.step(-1)

        @v.bind_key("f", overwrite=True)
        def fit(_):
            self._fit_view()

    def _toggle_layer_visibility(self, name: str) -> None:
        """Toggle exactly one layer, matching its eye control in the list."""
        layer = self.viewer.layers[name]
        layer.visible = not layer.visible

    def _install_trackpad_zoom(self) -> None:
        """Handle macOS native pinch events directly on the image canvas.

        Vispy normally translates these events itself, but Qt/Vispy backend
        combinations can silently drop them. Intercepting only the native zoom
        event avoids double handling and leaves wheel zoom and other gestures
        unchanged.
        """
        from qtpy.QtCore import QEvent, QObject, Qt

        viewer = self.viewer

        class _NativePinchFilter(QObject):
            def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
                if (
                    event.type() == QEvent.Type.NativeGesture
                    and event.gestureType()
                    == Qt.NativeGestureType.ZoomNativeGesture
                ):
                    _apply_pinch_zoom(viewer, event.value())
                    event.accept()
                    return True
                return super().eventFilter(watched, event)

        canvas = self.viewer.window._qt_viewer.canvas.native
        self._pinch_filter = _NativePinchFilter(canvas)
        canvas.installEventFilter(self._pinch_filter)

    def _schedule_initial_fit(self) -> None:
        """Fit after Qt has laid out the canvas and right-side docks."""
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, self._fit_view)

    def _fit_view(self) -> None:
        _fit_viewer(self.viewer)

    def _set_channel_view(self, mode: str) -> None:
        """Apply the controlled Hoechst / Calcein / merged channel view."""
        if mode not in CHANNEL_VIEWS:
            raise ValueError(f"unknown channel view: {mode}")
        self._channel_view = mode
        shown = {
            "total": mode in {VIEW_HOECHST, VIEW_MERGED},
            "alive": mode in {VIEW_CALCEIN, VIEW_MERGED},
        }
        for ch in CHANNELS:
            self.images[ch].visible = shown[ch]
            self.images[ch].colormap = IMAGE_CMAP[ch]
            self.images[ch].blending = "additive"
            self.images[ch].opacity = 1.0
            self.images[ch].gamma = 1.0
            self.images[ch].interpolation2d = "nearest"
            self.points[ch].visible = self._points_visible and shown[ch]
            self.points[ch].opacity = 1.0
            self.points[ch].blending = "translucent_no_depth"
        buttons = getattr(self, "channel_buttons", {})
        if mode in buttons:
            buttons[mode].setChecked(True)

    def _activate_pan_zoom(self) -> None:
        self._set_manual_mode_button("pan")
        for layer in self.points.values():
            layer.mode = "pan_zoom"

    def _activate_point_mode(self, channel: str, mode: str) -> None:
        """Select a marker layer and enter Napari's add or select mode."""
        self._set_manual_mode_button(f"{mode}_{channel}")
        self._points_visible = True
        self.points[channel].visible = True
        self.viewer.layers.selection.active = self.points[channel]
        self.points[channel].mode = mode

    def _set_manual_mode_button(self, mode: str) -> None:
        """Synchronise the highlighted manual tool with the active layer mode."""
        for key, button in getattr(self, "manual_mode_buttons", {}).items():
            button.native.setChecked(key == mode)

    def _toggle_grid(self) -> None:
        """Side-by-side view. Needs images and points interleaved, so reorder."""
        try:
            grid = self.viewer.grid
            if grid.enabled:
                grid.enabled = False
                self._order_for_overlay()
            else:
                self._order_for_grid()
                grid.stride = 2
                grid.enabled = True
        except Exception as exc:                      # pragma: no cover - GUI only
            self._set_warning(f"grid mode unavailable: {exc}")

    def _order_for_overlay(self) -> None:
        """Restore the requested top-to-bottom order in the layer list."""
        self._reorder_top_to_bottom(LAYER_LIST_TOP_TO_BOTTOM)

    def _order_for_grid(self) -> None:
        """Keep each image adjacent to its overlay in the requested order."""
        self._reorder_top_to_bottom(LAYER_LIST_TOP_TO_BOTTOM)

    def _reorder_top_to_bottom(self, names: tuple[str, ...]) -> None:
        """Translate visible layer-list order to Napari's reversed model."""
        self._reorder(list(reversed(names)))

    def _reorder(self, names: list[str]) -> None:
        for target, name in enumerate(names):
            current = self.viewer.layers.index(self.viewer.layers[name])
            if current != target:
                self.viewer.layers.move(current, target)

    # --- loading ---------------------------------------------------------

    def _load_data(self, name: str) -> FileData:
        settings = self.state[name]
        return detect.load_file(self.folder / name, settings.total.min_distance)

    def _get(self, name: str) -> FileData:
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            future = self._pending.pop(name, None)
        if future is not None:
            data = future.result()
        else:
            data = self._load_data(name)
        with self._lock:
            self._cache[name] = data
            if len(self._cache) > 3:                  # keep memory bounded
                for key in list(self._cache)[:-3]:
                    self._cache.pop(key, None)
        return data

    def _prefetch(self, index: int) -> None:
        if not 0 <= index < len(self.filenames):
            return
        name = self.filenames[index]
        with self._lock:
            if name in self._cache or name in self._pending:
                return
            self._pending[name] = self._pool.submit(self._load_data, name)

    def load(self, index: int) -> None:
        self.index = max(0, min(index, len(self.filenames) - 1))
        name = self.filenames[self.index]
        self.data = self._get(name)
        settings = self.state[name]

        self._updating = True
        try:
            for ch in CHANNELS:
                chan = getattr(self.data, ch)
                cset: ChannelSettings = getattr(settings, ch)
                layer = self.images[ch]
                layer.data = chan.image
                layer.contrast_limits_range = (
                    0.0, float(detect.SATURATION),
                )
                folder_lo, folder_hi = self._display_limits[ch]
                auto_low = float(np.clip(
                    detect.background_peak(chan.image),
                    folder_lo,
                    max(folder_lo, folder_hi - 1.0),
                ))
                self._auto_review_low[ch] = auto_low
                review_low = (
                    auto_low if cset.review_low is None
                    else float(np.clip(cset.review_low, folder_lo,
                                       max(folder_lo, folder_hi - 1.0)))
                )
                self.lut_sliders[ch].setValue(int(round(review_low)))
                self.lut_spinboxes[ch].setValue(int(round(review_low)))
                layer.contrast_limits = self._limits_for_channel(ch)
                for key, widget in self.sliders[ch].items():
                    widget.value = getattr(cset, key)
        finally:
            self._updating = False

        self._set_lut_controls_enabled(self._lut_mode == LUT_REVIEW)
        self._set_lut_status()
        self._set_channel_view(self._channel_view)
        self._draw_lut_histogram()

        self.refresh()
        # Replacing a Points layer's data makes Napari fall back to pan/zoom.
        # Reset our mode buttons after that replacement so the highlighted
        # button always describes what the next canvas interaction will do.
        self._activate_pan_zoom()
        self._prefetch(self.index + 1)

    # --- interaction -----------------------------------------------------

    def _current(self) -> tuple[str, object]:
        name = self.filenames[self.index]
        return name, self.state[name]

    def _settings(self, channel: str) -> ChannelSettings:
        return getattr(self._current()[1], channel)

    def _mark_current_pending(self) -> None:
        """Require acceptance again after a change that can alter counts."""
        _name, settings = self._current()
        settings.reviewed = False

    def _jump_choices(self) -> list[tuple[str, str]]:
        """(label, filename) pairs, set once; labels are refreshed in place."""
        return [(self._jump_label(i, n), n) for i, n in enumerate(self.filenames)]

    def _jump_label(self, index: int, name: str) -> str:
        return f"{'[x]' if self.state[name].reviewed else '[  ]'} {index + 1:2d}. {name}"

    def _refresh_jump_labels(self) -> None:
        """Update the tick marks without touching choices.

        Assigning to `choices` cannot be used here: magicgui diffs the list and
        moves any relabelled entry to the end, which would scramble the files
        out of filename order as they get reviewed. Setting the Qt item text
        directly keeps both the order and the label->filename mapping intact.
        """
        combo = self.w_jump.native
        for i, name in enumerate(self.filenames):
            label = self._jump_label(i, name)
            if combo.itemText(i) != label:
                combo.setItemText(i, label)

    def _on_jump(self, value=None) -> None:
        if self._updating:
            return
        name = self.w_jump.value
        if name is None:
            return
        index = self.filenames.index(name)
        if index != self.index:
            self.load(index)

    def _on_slider(self, channel: str) -> None:
        if self._updating:
            return
        cset = self._settings(channel)
        for key in ("threshold", "min_signal", "min_size", "max_size"):
            setattr(cset, key, self.sliders[channel][key].value)
        self._mark_current_pending()
        self.refresh()

    def _on_spacing(self, channel: str) -> None:
        """Spacing changes the candidate set, so this one re-runs detection."""
        if self._updating or self.data is None:
            return
        cset = self._settings(channel)
        cset.min_distance = int(self.sliders[channel]["min_distance"].value)
        chan = getattr(self.data, channel)
        z, signal = detect.compute_features(chan.image)
        chan.candidates = detect.find_candidates(
            z, cset.min_distance, self.data.um_per_px, signal
        )
        self._mark_current_pending()
        self.refresh()

    def _on_contrast(self, channel: str) -> None:
        """Reject uncontrolled Napari contrast changes.

        Generic layer controls are hidden, but this guard also protects against
        keyboard actions or plugins resetting contrast behind the controlled LUT
        panel.
        """
        if self._updating or self.data is None:
            return
        self._updating = True
        try:
            self.images[channel].contrast_limits = self._limits_for_channel(channel)
        finally:
            self._updating = False

    def _review_lower(self, channel: str) -> float:
        if hasattr(self, "lut_spinboxes"):
            return float(self.lut_spinboxes[channel].value())
        cset = self._settings(channel)
        return float(cset.review_low if cset.review_low is not None
                     else self._auto_review_low[channel])

    def _limits_for_channel(self, channel: str) -> tuple[float, float]:
        return _limits_for_display_mode(
            self._lut_mode,
            self._display_limits[channel],
            self._review_lower(channel),
            getattr(self.data, channel).image,
        )

    def _on_lut_lower(self, channel: str, value: int) -> None:
        if self._updating or self._lut_mode != LUT_REVIEW:
            return
        self._updating = True
        try:
            self.images[channel].contrast_limits = self._limits_for_channel(channel)
            self._settings(channel).review_low = float(value)
        finally:
            self._updating = False
        self.state.save()
        self._draw_lut_histogram()

    def _set_lut_controls_enabled(self, enabled: bool) -> None:
        for slider in self.lut_sliders.values():
            slider.setEnabled(enabled)
        for spin in self.lut_spinboxes.values():
            spin.setEnabled(enabled)
        self.b_reset_lut.setEnabled(enabled)

    def _set_lut_status(self) -> None:
        text = {
            LUT_REVIEW: (
                "REVIEW: lower black point only; folder upper limit is locked."
            ),
            LUT_COMPARE: (
                "COMPARE: the same folder-wide raw LUT is locked for every slide."
            ),
            LUT_RAW: "RAW: full native bit-depth; no contrast stretch.",
        }[self._lut_mode]
        self.w_lut_mode.setText(text)

    def _apply_lut_mode(self, mode: str) -> None:
        if mode not in LUT_MODES:
            raise ValueError(f"unknown display mode: {mode}")
        self._lut_mode = mode
        buttons = getattr(self, "lut_buttons", {})
        if mode in buttons:
            buttons[mode].setChecked(True)
        self._set_lut_controls_enabled(mode == LUT_REVIEW)
        self._set_lut_status()
        if self.data is None:
            return
        self._updating = True
        try:
            for ch in CHANNELS:
                self.images[ch].contrast_limits = self._limits_for_channel(ch)
        finally:
            self._updating = False
        self._draw_lut_histogram()

    def _reset_lut(self, value=None) -> None:
        """Return REVIEW lower limits to this slide's histogram peaks."""
        if self._updating or self.data is None:
            return
        self._lut_mode = LUT_REVIEW
        if LUT_REVIEW in self.lut_buttons:
            self.lut_buttons[LUT_REVIEW].setChecked(True)
        self._updating = True
        try:
            for ch in CHANNELS:
                value = int(round(self._auto_review_low[ch]))
                self.lut_sliders[ch].setValue(value)
                self.lut_spinboxes[ch].setValue(value)
                self._settings(ch).review_low = None
                self.images[ch].contrast_limits = self._limits_for_channel(ch)
        finally:
            self._updating = False
        self._set_lut_controls_enabled(True)
        self._set_lut_status()
        self.state.save()
        self._draw_lut_histogram()

    def _draw_lut_histogram(self) -> None:
        """Debounce waveform redraws while a lower-limit slider is dragged."""
        timer = getattr(self, "_lut_hist_timer", None)
        if timer is not None:
            timer.start()
            return
        self._draw_lut_histogram_now()

    def _draw_lut_histogram_now(self) -> None:
        if self.data is None or not hasattr(self, "lut_hist"):
            return
        limits = {ch: self._limits_for_channel(ch) for ch in CHANNELS}
        self.lut_hist.draw(self.data, limits, self._auto_review_low)

    def _on_points_edited(self, channel: str, event=None) -> None:
        """Translate a completed user edit into a persistent add or remove.

        Napari emits ADDING/REMOVING/CHANGING before it mutates the points
        arrays, then ADDED/REMOVED/CHANGED after completion. Refreshing from a
        pre-change callback re-enters the layer halfway through its mutation and
        can leave its view indices out of sync with ``data``.
        """
        if self._updating or self.data is None:
            return
        action = getattr(event, "action", None)
        action_name = getattr(action, "name", str(action)).upper().rsplit(".", 1)[-1]
        if action_name in {"ADDING", "REMOVING", "CHANGING"}:
            return
        cset = self._settings(channel)
        shown = _as_points(self.points[channel].data)
        expected = self._expected.get(channel, np.zeros((0, 2)))

        added = shown[~_match(shown, expected)]
        removed = expected[~_match(expected, shown)]

        for pt in added:
            cset.manual_add.append([float(pt[0]), float(pt[1])])

        for pt in removed:
            arr = np.asarray(cset.manual_add, dtype=float).reshape(-1, 2)
            hit = _match(arr, pt.reshape(1, 2)) if len(arr) else np.zeros(0, bool)
            if hit.any():
                # Undoing a point the user added earlier, not rejecting a real one.
                keep = [p for p, h in zip(cset.manual_add, hit) if not h]
                cset.manual_add = keep
            else:
                cset.manual_remove.append([float(pt[0]), float(pt[1])])

        if len(added) or len(removed):
            self._mark_current_pending()
        self.refresh()

    # --- rendering -------------------------------------------------------

    def counts(self) -> dict[str, tuple[np.ndarray, int]]:
        _name, settings = self._current()
        out = {}
        for ch in CHANNELS:
            cset: ChannelSettings = getattr(settings, ch)
            cands = getattr(self.data, ch).candidates
            out[ch] = detect.count(
                cands, cset.threshold, cset.min_size, cset.max_size,
                np.asarray(cset.manual_add, dtype=float).reshape(-1, 2),
                np.asarray(cset.manual_remove, dtype=float).reshape(-1, 2),
                min_signal=cset.min_signal,
            )

        total_points = out["total"][0]
        calcein_points = out["alive"][0]
        association = detect.associate_alive_to_nuclei(
            total_points,
            calcein_points,
            radius=detect.ALIVE_ASSOCIATION_RADIUS,
        )
        self._last_association = association
        alive_points = calcein_points[association.alive_indices]
        out["alive"] = (alive_points, len(alive_points))
        return out

    def refresh(self) -> None:
        if self.data is None:
            return
        name, settings = self._current()
        result = self.counts()

        self._updating = True
        try:
            for ch in CHANNELS:
                points, _ = result[ch]
                self._expected[ch] = points.copy()
                layer = self.points[ch]
                _replace_points(layer, points, self._marker_sizes(ch, points))

            # Relabel only when a review flag actually changed.
            sig = tuple(self.state[n].reviewed for n in self.filenames)
            if sig != self._jump_sig:
                self._jump_sig = sig
                self._refresh_jump_labels()
            self.w_jump.value = name
        finally:
            self._updating = False

        n_total, n_alive = result["total"][1], result["alive"][1]
        via = detect.viability(n_total, n_alive)
        settings.total_count, settings.alive_count = n_total, n_alive

        mark = "reviewed" if settings.reviewed else "pending"
        self.w_file.value = (
            f"{name}  [{self.index + 1}/{len(self.filenames)}]  {mark}"
        )
        self.w_progress.value = (
            f"{self.state.n_reviewed}/{self.state.n_files} reviewed"
        )
        self.w_counts.value = (
            f"Total {n_total}   Alive {n_alive}   "
            f"Viability {'--' if via is None else f'{via:.1f}%'}"
        )
        self._set_warning(self._warning_text(settings))
        self.b_export.enabled = self.state.all_reviewed()

        self.hist.draw(self.data, settings)
        self.state.save()

    def _marker_sizes(self, channel: str, points: np.ndarray) -> np.ndarray:
        """Scale each marker to its measured diameter so the size gate is visible."""
        cands = getattr(self.data, channel).candidates
        if not len(points):
            return np.zeros(0, dtype=float)
        sizes = np.full(len(points), 14.0)
        if len(cands):
            d2 = ((points[:, None, :] - cands.yx[None, :, :]) ** 2).sum(axis=2)
            near = d2.min(axis=1) <= MATCH_RADIUS**2
            idx = d2.argmin(axis=1)
            diam_px = cands.diameter[idx] / self.data.um_per_px
            sizes[near] = np.clip(diam_px[near], 6.0, 120.0)
        return sizes

    def _warning_text(self, settings) -> str:
        bits = []
        if self.data.warnings:
            bits.append("; ".join(self.data.warnings))
        for ch in CHANNELS:
            chan = getattr(self.data, ch)
            if chan.saturated_frac > 0.001:
                bits.append(f"{CHANNEL_LABEL[ch]}: {chan.saturated_frac:.2%} saturated")
            n_edits = getattr(settings, ch).n_manual_edits
            if n_edits:
                bits.append(f"{CHANNEL_LABEL[ch]}: {n_edits} manual edit(s)")
        if self._last_association is not None:
            unmatched = len(self._last_association.unmatched_alive_indices)
            if unmatched:
                bits.append(
                    f"Calcein: {unmatched} qualifying candidate(s) had no unique "
                    f"Hoechst match within {detect.ALIVE_ASSOCIATION_RADIUS:g} px"
                )
        return "\n".join(bits)

    def _set_warning(self, text: str) -> None:
        self.w_warn.value = text

    # --- navigation and export -------------------------------------------

    def step(self, delta: int) -> None:
        self.load(self.index + delta)

    def accept_and_next(self) -> None:
        name, settings = self._current()
        settings.reviewed = True
        self.state.save()
        nxt = self.state.next_unreviewed(after=name)
        if nxt is None:
            self.refresh()
            self._set_warning("All files reviewed - Export CSV is now enabled.")
            return
        self.load(self.filenames.index(nxt))

    def _copy_forward(self) -> None:
        name, _ = self._current()
        n = self.state.copy_settings_forward(name)
        self._set_warning(f"Settings copied to {n} remaining file(s).")

    def _reset_file(self) -> None:
        name, settings = self._current()
        for ch in CHANNELS:
            setattr(settings, ch, ChannelSettings())
        settings.reviewed = False
        self.state.save()
        self._cache.pop(name, None)
        self.load(self.index)

    def _export(self) -> None:
        try:
            per_file, per_group = export.export(self.state)
        except export.NotReviewedError as exc:
            self._set_warning(str(exc))
            return
        self._set_warning(f"Wrote {per_file.name} and {per_group.name}")


class _LUTHistogram:
    """Raw-pixel waveform with the controlled display limits overlaid."""

    def __init__(self):
        self.widget = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(3.6, 2.3), layout="constrained")
            self.widget = FigureCanvasQTAgg(self.figure)
            self.axes = self.figure.subplots(2, 1)
            self.figure.patch.set_alpha(0.0)
        except Exception:                             # pragma: no cover - GUI only
            self.figure = None

    def draw(
        self,
        data: FileData,
        limits: dict[str, tuple[float, float]],
        auto_lows: dict[str, float],
    ) -> None:
        if self.widget is None:
            return
        try:
            for ax, ch in zip(self.axes, CHANNELS):
                image = getattr(data, ch).image
                lo, hi = limits[ch]
                x_max = max(1.0, min(float(detect.SATURATION), hi))
                sample = image[::8, ::8].ravel()
                ax.clear()
                ax.hist(
                    sample,
                    bins=256,
                    range=(0.0, x_max),
                    color=IMAGE_CMAP[ch],
                    alpha=0.75,
                    log=True,
                )
                ax.axvline(auto_lows.get(ch, lo), color="#f0a030", lw=1.0,
                           ls="--")
                ax.axvline(lo, color="red", lw=1.2)
                ax.axvline(hi, color="#dddddd", lw=1.0, ls=":")
                ax.set_xlim(0.0, x_max)
                ax.set_ylabel(CHANNEL_LABEL[ch].split()[0], fontsize=8,
                              color="#cccccc")
                ax.tick_params(labelsize=7, colors="#cccccc")
                ax.set_facecolor("none")
                for spine in ax.spines.values():
                    spine.set_color("#666666")
            self.axes[-1].set_xlabel(
                "raw intensity  (red=current low, orange=auto peak)",
                fontsize=7,
                color="#cccccc",
            )
            self.widget.draw_idle()
        except Exception:                             # pragma: no cover - GUI only
            pass


class _Histogram:
    """Size distribution with the min/max gate drawn on it.

    Optional: if the matplotlib Qt backend is unavailable the app still runs,
    just without the plot.
    """

    def __init__(self):
        self.widget = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            # constrained_layout copes with the short dock height that
            # tight_layout warns about.
            self.figure = Figure(figsize=(3.6, 2.6), layout="constrained")
            self.widget = FigureCanvasQTAgg(self.figure)
            self.axes = self.figure.subplots(2, 1, sharex=True)
            self.figure.patch.set_alpha(0.0)
        except Exception:                             # pragma: no cover - GUI only
            self.figure = None

    def draw(self, data: FileData, settings) -> None:
        if self.widget is None:
            return
        try:
            for ax, ch in zip(self.axes, CHANNELS):
                cset = getattr(settings, ch)
                diam = getattr(data, ch).candidates.diameter
                ax.clear()
                if len(diam):
                    ax.hist(diam, bins=40, range=(0, 120),
                            color=POINT_COLOR[ch], alpha=0.75)
                ax.axvline(cset.min_size, color="red", lw=1.2)
                ax.axvline(cset.max_size, color="red", lw=1.2)
                ax.set_ylabel(CHANNEL_LABEL[ch].split()[0], fontsize=8,
                              color="#cccccc")
                ax.tick_params(labelsize=7, colors="#cccccc")
                ax.set_facecolor("none")
                for spine in ax.spines.values():
                    spine.set_color("#666666")
            self.axes[-1].set_xlabel("cell diameter (um)", fontsize=8,
                                     color="#cccccc")
            self.widget.draw_idle()
        except Exception:                             # pragma: no cover - GUI only
            pass


def main(folder: str | Path | None = None) -> None:
    import napari

    counter = CellCounter(Path(folder) if folder else Path.cwd())
    napari.run()


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)

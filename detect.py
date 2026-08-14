"""Loading and cell detection for two-channel Hoechst / Calcein AM TIFFs.

Pure functions, no GUI, so everything here can be tested headlessly.

Pipeline per channel:
    raw -> subtract Gaussian background -> smooth -> MAD z-score + raw signal
        -> peak_local_max candidates -> watershed sizing
Each candidate carries a score (z at its peak), its background-subtracted raw
signal, and a diameter (um). Counting is then a pure filter over those arrays,
followed by a unique Calcein-to-Hoechst association within 30 px. Candidate
score, signal, and size sliders filter stored measurements immediately; changing
peak spacing deliberately reruns candidate detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# --- acquisition defaults ---
# Only used when a file carries no PhysicalSizeX.  One micrometre per pixel is a
# neutral placeholder, not an inferred calibration; the loader warns whenever it
# must use it so size-gate results are not mistaken for calibrated measurements.
DEFAULT_UM_PER_PX = 1.0

# Common OME channel-name fragments.  Matching is case-insensitive; wavelength
# fragments retain compatibility with microscopes that store only filter names.
HOECHST_TAGS = ("hoechst", "385")
CALCEIN_TAGS = ("calcein", "470")
SATURATION = 65535         # uint16 clipping level
TIFF_SUFFIXES = frozenset({".tif", ".tiff"})

# --- detection constants ---
BG_SIGMA = 25.0            # background estimate; >> cell radius
SMOOTH_SIGMA = 2.0         # noise suppression before peak finding
BASE_Z = 3.0               # permissive level candidates are detected at
BASE_MIN_DISTANCE = 8      # px; ~half a cell diameter

# --- defaults deliberately wide; the review pass is where they get tightened ---
DEFAULT_THRESHOLD = 10.0   # z score
DEFAULT_MIN_SIGNAL = 500.0  # background-subtracted camera counts
DEFAULT_MIN_SIZE = 10.0    # um
DEFAULT_MAX_SIZE = 60.0    # um
ALIVE_ASSOCIATION_RADIUS = 30.0  # px; Calcein candidate to Hoechst nucleus


class ChannelError(RuntimeError):
    """Raised when a file does not expose the expected two channels."""


@dataclass
class Candidates:
    """Detected candidate cells for one channel, before any filtering."""

    yx: np.ndarray        # (N, 2) float, row/col in pixels
    score: np.ndarray     # (N,) float, z-score at the peak
    signal: np.ndarray    # (N,) float, raw peak height above local background
    diameter: np.ndarray  # (N,) float, equivalent diameter in um

    def __len__(self) -> int:
        return int(self.yx.shape[0])

    def mask(
        self,
        threshold: float,
        min_size: float,
        max_size: float,
        min_signal: float = DEFAULT_MIN_SIGNAL,
    ) -> np.ndarray:
        """Boolean mask of candidates passing score, signal, and size gates."""
        return (
            (self.score >= threshold)
            & (self.signal >= min_signal)
            & (self.diameter >= min_size)
            & (self.diameter <= max_size)
        )


@dataclass
class ChannelData:
    """Everything the GUI needs for one channel of one file."""

    image: np.ndarray          # raw uint16
    candidates: Candidates
    saturated_frac: float
    display_limits: tuple[float, float]  # sensible initial contrast window


@dataclass
class FileData:
    path: Path
    total: ChannelData         # Hoechst
    alive: ChannelData         # Calcein AM
    um_per_px: float = DEFAULT_UM_PER_PX
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Associations:
    """Unique Calcein-to-Hoechst assignments.

    ``alive_indices`` and ``total_indices`` are parallel arrays into the input
    point sets.  Every index appears at most once, so one nucleus contributes
    at most one live-cell count and duplicate Calcein peaks cannot inflate the
    result.
    """

    alive_indices: np.ndarray
    total_indices: np.ndarray
    distances: np.ndarray
    unmatched_alive_indices: np.ndarray


def channel_names(path: Path) -> list[str]:
    """Channel names from the OME-XML, in page order."""
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    return re.findall(r'<Channel[^>]*Name="([^"]+)"', xml)


def pixel_size(path: Path) -> tuple[float, str | None]:
    """(um per pixel, warning). Read per file so magnification can differ."""
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml)
    if match:
        try:
            value = float(match.group(1))
            if value > 0:
                return value, None
        except ValueError:
            pass
    return (
        DEFAULT_UM_PER_PX,
        f"no pixel calibration in metadata; assuming {DEFAULT_UM_PER_PX} um/px, "
        f"so sizes in um may be wrong",
    )


def load_channels(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (hoechst, calcein, warnings), selecting pages by OME channel name.

    Selecting by name rather than index tolerates extra pages and inconsistent
    page declarations; the first occurrence of each recognised channel is used.
    """
    warnings: list[str] = []
    with tifffile.TiffFile(path) as tf:
        pages = [p.asarray() for p in tf.pages]
        xml = tf.ome_metadata or ""

    names = re.findall(r'<Channel[^>]*Name="([^"]+)"', xml)

    if len(names) < len(pages):
        # Fall back to page order if the XML is short or missing.
        warnings.append(
            f"{len(pages)} pages but {len(names)} channel names; using page order"
        )
        names = names + [""] * (len(pages) - len(names))
    elif len(names) > len(pages):
        # Read real IFD pages rather than allowing surplus OME declarations to
        # create synthetic empty frames.
        warnings.append(
            f"OME-XML declares {len(names)} channels but only {len(pages)} image "
            f"frames exist; ignoring the surplus declaration"
        )

    def first_with(tags: tuple[str, ...]) -> int | None:
        for i, n in enumerate(names[: len(pages)]):
            folded = n.casefold()
            if any(tag.casefold() in folded for tag in tags):
                return i
        return None

    i_tot, i_alive = first_with(HOECHST_TAGS), first_with(CALCEIN_TAGS)

    if i_tot is None or i_alive is None:
        if len(pages) >= 2:
            warnings.append(
                "could not identify channels by name; assuming page 0 = Hoechst, "
                "page 1 = Calcein"
            )
            i_tot, i_alive = 0, 1
        else:
            raise ChannelError(f"{path.name}: need 2 channels, found {len(pages)} page(s)")

    if len(pages) > 2:
        warnings.append(f"{len(pages)} pages present; using pages {i_tot} and {i_alive}")

    return pages[i_tot], pages[i_alive], warnings


def discover_tiffs(folder: Path) -> list[Path]:
    """Image files accepted by the application, independent of suffix case."""
    return [
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.casefold() in TIFF_SUFFIXES
    ]


def compute_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return noise-normalised z score and raw signal-above-background images.

    The Gaussian background removes the strong illumination gradient in the
    calcein channel and the dim out-of-focus halos in both, so a single
    threshold becomes meaningful across the whole field. MAD normalisation then
    puts different exposures on a roughly comparable scale. The unnormalised
    signal is retained as a second gate: without it, weak camera noise in a
    nearly blank frame can receive a deceptively high z score merely because
    that frame has a very small MAD.
    """
    img = image.astype(np.float32)
    flat = img - ndi.gaussian_filter(img, BG_SIGMA)
    smooth = ndi.gaussian_filter(flat, SMOOTH_SIGMA)
    median = float(np.median(smooth))
    signal = smooth - median
    mad = float(np.median(np.abs(smooth - median))) * 1.4826
    if mad <= 0:
        mad = 1.0
    return signal / mad, signal


def compute_z(image: np.ndarray) -> np.ndarray:
    """Compatibility helper returning only the noise-normalised score image."""
    return compute_features(image)[0]


def find_candidates(
    z: np.ndarray,
    min_distance: int = BASE_MIN_DISTANCE,
    um_per_px: float = DEFAULT_UM_PER_PX,
    signal: np.ndarray | None = None,
) -> Candidates:
    """Detect candidates permissively and measure each one's size.

    Size comes from a watershed of the z >= BASE_Z mask seeded on the peaks, so
    touching cells are split apart and every candidate gets an area even in a
    clump. Measuring at a fixed permissive level (rather than at the user's
    threshold) keeps diameter a stable property: moving the threshold slider
    changes which cells are counted, never how big they are measured to be.
    """
    peaks = peak_local_max(z, min_distance=min_distance, threshold_abs=BASE_Z)
    if peaks.size == 0:
        empty_i = np.zeros((0, 2), dtype=float)
        empty_f = np.zeros((0,), dtype=float)
        return Candidates(empty_i, empty_f, empty_f, empty_f)

    scores = z[peaks[:, 0], peaks[:, 1]].astype(float)
    # Callers that only have a z image keep the historical behaviour (no raw
    # gate). The normal loading path always supplies the real signal image.
    signals = (
        signal[peaks[:, 0], peaks[:, 1]].astype(float)
        if signal is not None
        else np.full(len(peaks), np.inf, dtype=float)
    )

    markers = np.zeros(z.shape, dtype=np.int32)
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
    labels = watershed(-z, markers, mask=z >= BASE_Z)

    areas = np.bincount(labels.ravel(), minlength=len(peaks) + 1)[1:]
    diameter = 2.0 * np.sqrt(areas / np.pi) * um_per_px

    return Candidates(
        peaks.astype(float), scores, signals, diameter.astype(float)
    )


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    """Initial contrast window: robust percentiles, so dots are visible at once."""
    lo, hi = np.percentile(image, [1.0, 99.9])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def background_peak(image: np.ndarray, sample_stride: int = 8) -> float:
    """Robust estimate of the dominant raw-intensity background peak.

    A fixed-width 16-bit histogram is used rather than an exact integer mode:
    camera noise spreads a real background population over many neighbouring
    values, so the exact most-common value is unstable.  Mild histogram
    smoothing finds the centre of that population without following isolated
    cell peaks in the bright tail.  This value is for the display black point
    only and never enters detection or counting.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0
    stride = max(1, int(sample_stride))
    sample = arr[::stride, ::stride].ravel()
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        lo, hi = float(info.min), float(info.max) + 1.0
        bins = 1024 if info.bits >= 16 else min(256, int(hi - lo))
    else:
        lo, hi = float(np.nanmin(sample)), float(np.nanmax(sample))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return lo if np.isfinite(lo) else 0.0
        bins = 1024
    hist, edges = np.histogram(sample, bins=bins, range=(lo, hi))
    smooth = ndi.gaussian_filter1d(hist.astype(float), sigma=1.5)
    index = int(np.argmax(smooth))
    return float((edges[index] + edges[index + 1]) / 2.0)


def shared_display_limits(
    paths: list[Path], sample_stride: int = 16
) -> dict[str, tuple[float, float]]:
    """One robust raw-intensity window per channel for an entire folder.

    Per-image auto contrast makes a nearly blank calcein frame look as bright
    as a strongly fluorescent one. A shared window preserves cross-file
    intensity differences while remaining much more useful than displaying the
    entire uint16 range. Sampling keeps startup fast and memory bounded.
    """
    sampled: dict[str, list[np.ndarray]] = {"total": [], "alive": []}
    for path in paths:
        total, alive, _ = load_channels(Path(path))
        sampled["total"].append(total[::sample_stride, ::sample_stride].ravel())
        sampled["alive"].append(alive[::sample_stride, ::sample_stride].ravel())

    limits: dict[str, tuple[float, float]] = {}
    for channel, chunks in sampled.items():
        if not chunks:
            limits[channel] = (0.0, float(SATURATION))
            continue
        limits[channel] = _display_limits(np.concatenate(chunks))
    return limits


def _channel_data(
    image: np.ndarray, min_distance: int, um_per_px: float
) -> ChannelData:
    z, signal = compute_features(image)
    return ChannelData(
        image=image,
        candidates=find_candidates(z, min_distance, um_per_px, signal),
        saturated_frac=float((image >= SATURATION).mean()),
        display_limits=_display_limits(image),
    )


def load_file(path: Path, min_distance: int = BASE_MIN_DISTANCE) -> FileData:
    """Load one TIFF and run detection on both channels."""
    path = Path(path)
    hoechst, calcein, warnings = load_channels(path)
    um_per_px, cal_warning = pixel_size(path)
    if cal_warning:
        warnings.append(cal_warning)
    return FileData(
        path=path,
        total=_channel_data(hoechst, min_distance, um_per_px),
        alive=_channel_data(calcein, min_distance, um_per_px),
        um_per_px=um_per_px,
        warnings=warnings,
    )


def count(
    candidates: Candidates,
    threshold: float,
    min_size: float,
    max_size: float,
    manual_add: np.ndarray | None = None,
    manual_remove: np.ndarray | None = None,
    min_signal: float = DEFAULT_MIN_SIGNAL,
) -> tuple[np.ndarray, int]:
    """Points to display and the resulting count.

    Manual edits are applied on top of the automatic result: removals suppress
    an auto candidate near the clicked spot, additions are always kept. Both are
    stored as coordinates rather than indices, so they survive slider changes.
    """
    keep = candidates.mask(threshold, min_size, max_size, min_signal)
    points = candidates.yx[keep]

    if manual_remove is not None and len(manual_remove) and len(points):
        points = _drop_near(points, np.asarray(manual_remove, dtype=float))

    if manual_add is not None and len(manual_add):
        extra = np.asarray(manual_add, dtype=float).reshape(-1, 2)
        points = np.vstack([points, extra]) if len(points) else extra

    return points, int(len(points))


def associate_alive_to_nuclei(
    total_yx: np.ndarray,
    alive_yx: np.ndarray,
    radius: float = ALIVE_ASSOCIATION_RADIUS,
) -> Associations:
    """Pair Calcein candidates uniquely to Hoechst nuclei within ``radius``.

    The sparse assignment has two priorities, in order:

    1. maximise the number of unique nucleus-Calcein pairs;
    2. minimise their total distance.

    A private high-cost dummy target lets every Calcein point remain unmatched.
    Its cost is larger than every possible change in real-edge distance, making
    cardinality the primary objective.  This is preferable to a greedy nearest
    neighbour pass, which can give two Calcein peaks to one nucleus while
    needlessly leaving an adjacent nucleus unmatched.
    """
    total = np.asarray(total_yx, dtype=float).reshape(-1, 2)
    alive = np.asarray(alive_yx, dtype=float).reshape(-1, 2)
    n_total, n_alive = len(total), len(alive)

    empty_i = np.zeros(0, dtype=int)
    empty_f = np.zeros(0, dtype=float)
    if n_alive == 0:
        return Associations(empty_i, empty_i, empty_f, empty_i)
    if n_total == 0 or radius <= 0:
        return Associations(
            empty_i, empty_i, empty_f, np.arange(n_alive, dtype=int)
        )

    tree = cKDTree(total)
    neighbours = tree.query_ball_point(alive, r=float(radius))

    rows: list[int] = []
    cols: list[int] = []
    costs: list[float] = []
    for alive_idx, total_indices in enumerate(neighbours):
        if total_indices:
            indices = np.asarray(total_indices, dtype=int)
            distances = np.linalg.norm(total[indices] - alive[alive_idx], axis=1)
            rows.extend([alive_idx] * len(indices))
            cols.extend(indices.tolist())
            # Sparse matrices discard explicit zeroes.  The epsilon preserves
            # exact-overlap edges without affecting any practical tie-break.
            costs.extend((distances + 1e-9).tolist())

    # One private dummy column per Calcein point guarantees a full row matching.
    # One fewer real match must cost more than changing every real edge from the
    # maximum radius to zero, so maximum cardinality dominates total distance.
    unmatched_cost = (n_alive + 1) * (float(radius) + 1.0)
    rows.extend(range(n_alive))
    cols.extend((n_total + i for i in range(n_alive)))
    costs.extend([unmatched_cost] * n_alive)

    graph = csr_matrix(
        (np.asarray(costs, dtype=float), (rows, cols)),
        shape=(n_alive, n_total + n_alive),
    )
    row_ind, col_ind = min_weight_full_bipartite_matching(graph)
    real = col_ind < n_total
    alive_indices = row_ind[real].astype(int, copy=False)
    total_indices = col_ind[real].astype(int, copy=False)

    # The scipy result is normally row-sorted; make that guarantee explicit so
    # rendered point order and tests remain deterministic across scipy versions.
    order = np.argsort(alive_indices, kind="stable")
    alive_indices = alive_indices[order]
    total_indices = total_indices[order]
    distances = np.linalg.norm(
        alive[alive_indices] - total[total_indices], axis=1
    )

    matched = np.zeros(n_alive, dtype=bool)
    matched[alive_indices] = True
    return Associations(
        alive_indices=alive_indices,
        total_indices=total_indices,
        distances=distances.astype(float, copy=False),
        unmatched_alive_indices=np.flatnonzero(~matched),
    )


def _drop_near(points: np.ndarray, removed: np.ndarray, radius: float = 12.0) -> np.ndarray:
    """Drop auto points within `radius` px of a manually removed location."""
    if not len(points):
        return points
    d2 = ((points[:, None, :] - removed[None, :, :]) ** 2).sum(axis=2)
    return points[d2.min(axis=1) > radius**2]


def viability(total: int, alive: int) -> float | None:
    """Live fraction as a percentage; None when there is nothing to divide by."""
    if total <= 0:
        return None
    return 100.0 * alive / total

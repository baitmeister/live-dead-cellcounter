"""Render detection overlays to PNG for spot-checking without opening the GUI.

    python3 -m cellcounter.qc sample-a.tif sample-b.tiff --out qc_overlays

Markers are drawn at each cell's measured diameter, so the size gate is visible
rather than inferred. Score-detected candidates excluded by the raw-signal gate,
size gate, or nucleus association are drawn in dim red.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import detect
from .state import file_sort_key

KEPT_COLOR = {"total": (0, 255, 200), "alive": (255, 230, 0)}
REJECT_COLOR = (200, 40, 40)


def _grayscale(image: np.ndarray, dim: float = 0.8) -> Image.Image:
    lo, hi = np.percentile(image, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1
    g = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1) * dim
    return Image.fromarray((np.dstack([g, g, g]) * 255).astype(np.uint8))


def render_channel(
    chan: detect.ChannelData,
    which: str,
    threshold: float,
    min_signal: float,
    min_size: float,
    max_size: float,
    crop: tuple[int, int, int] | None = None,
    zoom: int = 1,
    show_rejected: bool = True,
    um_per_px: float = detect.DEFAULT_UM_PER_PX,
    accepted_yx: np.ndarray | None = None,
) -> Image.Image:
    """One channel with its overlay. `crop` is (row, col, size) in pixels."""
    image = chan.image
    y0, x0 = 0, 0
    if crop:
        y0, x0, size = crop
        image = image[y0:y0 + size, x0:x0 + size]

    img = _grayscale(image)
    if zoom != 1:
        img = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    cands = chan.candidates
    passes_score = cands.score >= threshold
    passes_signal = cands.signal >= min_signal
    passes_size = (cands.diameter >= min_size) & (cands.diameter <= max_size)
    passes_association = np.ones(len(cands), dtype=bool)
    if accepted_yx is not None:
        accepted = np.asarray(accepted_yx, dtype=float).reshape(-1, 2)
        if len(accepted):
            d2 = ((cands.yx[:, None, :] - accepted[None, :, :]) ** 2).sum(axis=2)
            passes_association = d2.min(axis=1) <= 0.25
        else:
            passes_association[:] = False

    for keep, color in (
        (passes_score & passes_signal & passes_size & passes_association,
         KEPT_COLOR[which]),
        (passes_score & ~(passes_signal & passes_size & passes_association),
         REJECT_COLOR)
        if show_rejected else (None, None),
    ):
        if keep is None:
            continue
        for (y, x), diam in zip(cands.yx[keep], cands.diameter[keep]):
            cy, cx = (y - y0) * zoom, (x - x0) * zoom
            if not (0 <= cx < img.width and 0 <= cy < img.height):
                continue
            r = max(3.0, diam / um_per_px / 2 * zoom)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)

    return img


def render_file(
    path: Path,
    out_dir: Path,
    threshold: float = detect.DEFAULT_THRESHOLD,
    min_signal: float = detect.DEFAULT_MIN_SIGNAL,
    min_size: float = detect.DEFAULT_MIN_SIZE,
    max_size: float = detect.DEFAULT_MAX_SIZE,
    crop: tuple[int, int, int] | None = None,
    zoom: int = 1,
) -> Path:
    """Side-by-side Hoechst | Calcein overlay for one file."""
    data = detect.load_file(path)
    total_points, n_total = detect.count(
        data.total.candidates, threshold, min_size, max_size,
        min_signal=min_signal,
    )
    calcein_points, _ = detect.count(
        data.alive.candidates, threshold, min_size, max_size,
        min_signal=min_signal,
    )
    association = detect.associate_alive_to_nuclei(total_points, calcein_points)
    alive_points = calcein_points[association.alive_indices]

    tiles = []
    for which in ("total", "alive"):
        chan = getattr(data, which)
        tiles.append(render_channel(chan, which, threshold, min_signal,
                                    min_size, max_size,
                                    crop, zoom, um_per_px=data.um_per_px,
                                    accepted_yx=(alive_points
                                                 if which == "alive" else None)))

    counts = [n_total, len(alive_points)]

    gap = 10
    combo = Image.new("RGB", (tiles[0].width * 2 + gap, tiles[0].height), (25, 25, 25))
    combo.paste(tiles[0], (0, 0))
    combo.paste(tiles[1], (tiles[0].width + gap, 0))

    via = detect.viability(*counts)
    caption = (f"{path.name}   total={counts[0]}  alive={counts[1]}  "
               f"viability={'--' if via is None else f'{via:.1f}%'}   "
               f"z>={threshold}  signal>={min_signal}  "
               f"size {min_size}-{max_size}um  "
               f"association<={detect.ALIVE_ASSOCIATION_RADIUS:g}px")
    ImageDraw.Draw(combo).text((8, 6), caption, fill=(255, 255, 255))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"qc_{path.stem}.png"
    combo.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="TIFFs to render (default: all)")
    ap.add_argument("--out", default="qc_overlays")
    ap.add_argument("--threshold", type=float, default=detect.DEFAULT_THRESHOLD)
    ap.add_argument("--min-signal", type=float, default=detect.DEFAULT_MIN_SIGNAL)
    ap.add_argument("--min-size", type=float, default=detect.DEFAULT_MIN_SIZE)
    ap.add_argument("--max-size", type=float, default=detect.DEFAULT_MAX_SIZE)
    ap.add_argument("--crop", nargs=3, type=int, metavar=("ROW", "COL", "SIZE"),
                    help="render a detail crop instead of the full field")
    ap.add_argument("--zoom", type=int, default=1)
    args = ap.parse_args()

    folder = Path.cwd()
    paths = ([folder / f for f in args.files] if args.files
             else sorted(detect.discover_tiffs(folder),
                         key=lambda p: file_sort_key(p.name)))

    for path in paths:
        out = render_file(path, folder / args.out, args.threshold,
                          args.min_signal, args.min_size, args.max_size,
                          tuple(args.crop) if args.crop else None, args.zoom)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

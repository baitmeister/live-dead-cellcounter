# Cell Viability Counter

Counts Hoechst-positive nuclei (total cells) and Calcein AM-positive cells (live
cells) in multi-page `.tif` images. Every image must pass an interactive review
before the program will export CSV results. The source TIFFs are never modified.

**Author:** Hao Li, with GPT-5.6 Sol  
**License:** [MIT License](LICENSE)

## Quick start

Keep the complete `cellcounter/` program folder inside the folder containing the
TIFF images:

```text
image-folder/
├── sample-a.tif
├── sample-b.tiff
└── cellcounter/
    ├── run_counter.command
    ├── app.py
    └── ...
```

Double-click **`cellcounter/run_counter.command`**. The launcher automatically
uses the folder immediately containing `cellcounter/` as the image folder. From
a terminal, the equivalent generic command is:

```bash
cd "/path/to/image-folder"
./cellcounter/run_counter.command
```

The program discovers `.tif` and `.tiff` files case-insensitively.

The first pending image opens centered with the complete field fitted in the
viewer. Hoechst is blue, Calcein is green, total-cell markers are cyan, and
live-cell markers are yellow. Each marker is drawn at the candidate's measured
diameter.

## Panel map

| Panel | Location | Purpose |
|---|---|---|
| **Display LUT** | Left | Channel view, Review/Compare/RAW display, background black points, and raw-intensity waveforms. |
| **Review** | Right | Current file, progress, counts, warnings, file jump, navigation, reset, and export. |
| **Detection & Manual** | Right | Detection settings and manual addition/removal tools. |
| **Size distribution** | Separate dock | Candidate-diameter histograms and the current minimum/maximum size gates. |

Napari's generic layer controls are hidden. This prevents per-slide
auto-contrast, gamma, opacity, blending, transform, or marker-style changes from
silently making one image look different from another. Zooming and panning remain
camera operations and do not alter the image or counts.

## Recommended review workflow

1. Inspect the image in **Merged**, then check **Hoechst** and **Calcein**
   separately.
2. Use **Compare** when comparing intensity between slides. Use **RAW** when you
   need to inspect the full native camera range.
3. In **Review**, adjust only the Hoechst or Calcein background black point if
   background obscures the cells. This changes display only.
4. Adjust detection thresholds and size limits while checking the markers,
   counts, warnings, and size distributions.
5. Correct missed or false detections with the manual tools. Pay particular
   attention to saturated fields and unassociated Calcein warnings.
6. Click **Accept & Next** only when the displayed count is acceptable. Any later
   count-changing slider, spacing, or manual edit marks that file pending again.
7. After every image is accepted, click **Export CSV**.

Progress is saved continuously to `review_state.json`, so the program can be
closed and resumed without losing settings or manual corrections.

## Display LUT controls

### Channel

The buttons are ordered **Hoechst / Calcein / Merged** from left to right. They
control which image and corresponding markers are visible.

### View

The buttons are ordered **Review / Compare / RAW**. These modes change only how
the original TIFF pixels are displayed; detection and exported counts always use
the original pixel values.

- **Review** uses the folder-wide upper display limit and lets you adjust only the
  lower black point. The lower point can never go below the folder-wide Compare
  lower limit, so Review cannot brighten pixels relative to Compare. Raising the
  lower point dims more background; lowering it reveals more background. Each
  slide starts at its dominant raw-intensity background histogram peak.
- **Compare** locks one folder-wide raw-intensity window per channel for every
  slide. It preserves meaningful brightness differences between slides.
- **RAW** displays the complete native data range with no contrast stretch. For
  example, the native range of a uint16 image is `0–65535`.

The **Hoechst background** and **Calcein background** sliders and number boxes are
enabled only in Review. **Auto-set background from histogram** returns both lower
black points to the current slide's automatically detected background peaks and
switches back to Review.

Each waveform uses a logarithmic count axis:

- orange dashed line: automatic background peak;
- red line: current lower black point;
- grey dotted line: locked upper display limit.

These LUT controls are for visual review only. Use `thresh z`, `min signal`, and
the size gates to change which cells count.

## Detection controls

Hoechst and Calcein have separate settings because their background and signal
distributions differ.

| Control | Effect |
|---|---|
| `thresh z` | Requires a candidate peak to rise a given number of robust noise units above its local background. Higher values reject weaker candidates; lower values include dimmer or noisier candidates. |
| `min signal` | Minimum background-subtracted peak height in raw camera counts. Raise it to reject weak fluctuations that acquire a high z-score in an otherwise quiet image. |
| `min um` | Rejects candidates below this measured diameter. |
| `max um` | Rejects candidates above this measured diameter, including many clumps and halos. |
| `spacing` | Minimum separation between candidate peaks in pixels. Raising it suppresses nearby peaks; changing it reruns candidate detection and may pause briefly. |

Threshold, minimum signal, and size changes filter measurements that were already
computed, so their previews are immediate. Spacing is different because it
changes where candidate peaks are found.

The **Size distribution** panel shows all permissively detected candidate
diameters. Red vertical lines show the active `min um` and `max um` gates.

## Nucleus-Calcein association

After both channels are filtered, the program pairs Calcein candidates uniquely
to Hoechst nuclei within a fixed **30 px** radius. The assignment:

1. maximises the number of unique nucleus-Calcein pairs; then
2. minimises the total distance across those pairs.

This is a global unique assignment, not a "strongest Calcein" rule and not a
simple greedy nearest-neighbour pass. One nucleus can contribute at most one live
cell, so duplicate Calcein peaks cannot inflate the automatic live count above
the total count. Qualifying Calcein candidates without a unique nucleus within
30 px are excluded and reported in the Review warning area.

## Manual corrections

- **Add total / Add alive:** select the appropriate button, then click the missed
  cell.
- **Select/remove total / Select/remove alive:** select the appropriate button,
  click a marker, then press `Delete` or `Backspace`.
- **Pan / zoom:** returns the mouse to navigation mode.

The active manual mode stays highlighted in blue. The highlight therefore shows
what the next canvas click will do; choose **Pan / zoom** when you finish editing.

A manually added live point is subject to the same 30 px unique-association rule
as an automatic Calcein candidate. If both the nucleus and Calcein signal were
missed, add the **total** nucleus first and then add the **alive** point. An alive
point without an available total nucleus in range is excluded from the displayed
live count and produces an association warning.

Manual edits are stored as coordinates and survive detection-slider changes.
Deleting a point that you manually added undoes that addition rather than
recording a separate rejection. Deleting or adding a total nucleus can also
change which Calcein candidates can be uniquely associated.

## Navigation and buttons

The **jump to** list remains in natural filename order, so `sample-2.tif` appears
before `sample-10.tif`. `[x]` means accepted and `[  ]` means pending. Jumping or
using **< Back** never accepts the current image.

- **Copy settings forward** — copies both channels' detection settings to every
  later pending file. It does not copy Review LUT values or manual edits.
- **Reset this file** — restores default detection settings, clears manual edits
  and per-file Review black points, and marks the image pending.
- **Fit image to window** — centers the TIFF and fits the complete field.
- **< Back** — opens the previous file without accepting the current one.
- **Accept & Next** — accepts the current counts and opens the next pending file.
- **Export CSV** — remains disabled until every file is accepted.

### Keyboard and trackpad

`1` Hoechst · `2` Calcein · `3` Merged · `Space` toggle markers · `g` grid ·
`f` center/fit · `n` accept and next · `b` back

Press `3` before `g` if you want both channels visible in the side-by-side grid.
Pinch-to-zoom on a macOS trackpad and mouse-wheel zoom are supported. Use `f` if
you need to return to the centered complete-field view.

## Output

Export writes two files next to the images:

- **`viability.csv`** — `group, replicate, file, total_cells, alive_cells, viability_pct`
- **`viability_by_group.csv`** — `group, n_replicates, mean_viability_pct, sd_viability_pct, sem_viability_pct`

Groups come from the final dot in each filename: `treatment-a.3.tif` becomes group
`treatment-a`, replicate `3`. A group with one valid replicate receives blank SD
and SEM fields rather than zero. An image with zero total cells receives a blank
viability value.

The program applies no negative-control, positive-control, or inter-slide
normalisation. It reports reviewed cell counts and viability using the same
detection pipeline and association rule for every slide.

## Using another image folder

The program is not tied to a particular image set. For the normal arrangement,
copy the complete `cellcounter/` folder into the image folder and leave
`run_counter.command` inside `cellcounter/`.

Alternatively, keep one program copy and pass another image folder explicitly:

```bash
/path/to/program-parent/cellcounter/run_counter.command "/path/to/other/image-folder"
```

`review_state.json` and exported CSVs are written next to the selected images,
not inside the `cellcounter/` program folder. This is true for both the automatic
parent-folder default and an explicitly supplied image folder.

What adapts automatically:

- **Filename count and grouping.** Names are parsed as `group.replicate`; a name
  without a dot becomes its own group.
- **Number of files.** Export unlocks after every discovered file is accepted.
- **Pixel calibration.** OME `PhysicalSizeX` is read from each file so the µm
  gates follow magnification. Missing calibration produces a warning and falls
  back to a neutral 1.0 µm/px placeholder. Do not interpret size-gated results as
  calibrated measurements until valid metadata is supplied.
- **Extra TIFF pages.** The loader selects the first Hoechst and Calcein pages by
  OME channel name and reports surplus pages or metadata inconsistencies.

Acquisition compatibility checklist:

- **Channel identity:** OME names are matched case-insensitively using the common
  Hoechst fragments `hoechst` or `385` and Calcein fragments `calcein` or `470`.
  If neither channel can be identified, the program warns and assumes page 0 =
  Hoechst and page 1 = Calcein. Verify the channel assignment before reviewing
  results. The aliases are defined by `HOECHST_TAGS` and `CALCEIN_TAGS` in
  `cellcounter/detect.py`.
- **Threshold and size defaults:** these are starting values, not universal
  biological cutoffs. Re-tune them on representative images from each acquisition
  setup.
- **Pixel-scale parameters:** peak spacing, the 25 px background radius, 2 px
  smoothing, and the 30 px nucleus-Calcein association radius are fixed pixel
  distances. OME calibration rescales the diameter gates, but it does not rescale
  these parameters. Validate them when magnification, camera binning, cell type,
  or acquisition geometry changes.
- **Filename grouping:** group summaries assume the final dot separates group and
  replicate. Rename files if a different naming convention would produce the
  wrong grouping.

## Checking without the GUI

Run the headless verification suite from the image folder:

```bash
cd "/path/to/image-folder"
python3 -m cellcounter.selftest
```

It checks channel identification, detection, size gating, unique association,
manual-edit behavior, review-state migration, pinch/fit helpers, and the export
gate, then prints default counts and viability for every file.

Create an overlay for visual spot-checking:

```bash
python3 -m cellcounter.qc sample-a.tif --crop 800 800 400 --zoom 2
```

The command writes PNGs to `qc_overlays/`. Cyan/yellow markers are counted
Hoechst/Calcein candidates. Dim red markers are score-detected candidates rejected
by minimum signal, size, or nucleus association.

## How detection works

For each channel, the program:

1. subtracts a Gaussian background estimate (σ = 25 px) to reduce illumination
   gradients and broad out-of-focus halos;
2. smooths the result (σ = 2 px);
3. normalises it using the median absolute deviation, producing the robust
   `thresh z` score;
4. retains the unnormalised background-subtracted peak height for `min signal`;
5. finds permissive local maxima and measures candidate diameter by watershed;
6. filters candidates using the channel's z, signal, and size settings; and
7. performs the unique 30 px nucleus-Calcein association.

Candidate score, signal, and diameter are stored, so most slider previews are
immediate. Changing spacing reruns peak finding and watershed sizing.

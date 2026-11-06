# SlicerMLArray rework for MLArray 1.0 — plan

Status: **approved and merged into `main` (`a0f0f8a`, pushed to GitLab 2026-11-06); next: GitHub mirror, Extensions Index** ( decisions in §8 confirmed, deviations from the plan in §9). Branch `rework/v1` (from `main` @ `d32e3f3`, which is 2 commits ahead
of `origin/main`). Work through the stages in order; tick a step and add its commit hash when it is done.

Background reading: `~/projects/mlarray/docs/design/rework-plan.md` ("Start here"), the MLArray v1 spec
(`~/projects/mlarray/docs/design/spec.md` §4 core, §5.2 `ml` profile), the orientation KB chapter on Slicer
(`~/projects/image-orientation-kb/docs/06-viewers.md` §3), and napari-mlarray as the reference consumer of v1
(`~/projects/napari-mlarray/src/napari_mlarray/`).

## 1. Why a rewrite

The current code (≈2.2k lines) is written against the **v0** API, which MLArray 1.0 removed: `meta.spatial`
(`spacing/origin/direction/affine/axis_labels/coord_system`), `meta.is_seg`, `meta.bbox`/`MetaBbox`,
`meta.extra`, `MetaSpatial`. Nothing of it loads a v1 file; the dependency check even rejects mlarray 1.0
(`MetaSpatial` missing). Its concepts also don't carry over:

- world handedness came from a stored letter code (`coord_system`), which v1 forbids; v1 states the direction of
  every space axis (`increases_toward`);
- array axes were recognised by magic labels (`spatial_x`…); v1 has typed, named axes and an explicit
  `from_array` matrix with named columns;
- segmentations were labelmaps without names or colours; v1 has a label table (`values.labels`, with codes);
- boxes were per-axis voxel pairs in "metadata order"; v1 boxes live in `profiles.ml.boxes` with a frame
  (`index` or a space) and named axes;
- export wrote RAS geometry and an `extra.slicer_mlarray` block; v1 recommends the DICOM patient frame (LPS)
  and keeps the source metadata untouched.

So: rewrite, keep only the Slicer plumbing that was right (scripted `FileReader`/`FileWriter` classes, on-demand
`pip_install`, ROI nodes for boxes). No v0 compatibility (same rule as MLArray itself).

## 2. Slicer facts the design relies on (KB ch06 §3, re-check on 5.12.4)

- Slicer holds geometry as **IJKToRAS** (4×4, any rotation/flip/permutation, no shear restriction beyond VTK's
  direction matrix) and `arrayFromVolume` is numpy `[k, j, i]`. So a C-order MLArray array `[a0, a1, a2]` is
  used **as is**: I = a2, J = a1, K = a0. No data reordering, no resampling (unlike napari).
- Showing a volume snaps the slice views to the volume's own axes ("Reset view orientation on show"), so
  oblique grids are displayed in their acquisition planes. Nothing to do on our side.
- Slicer's views are radiological by default and its world is RAS; files are LPS. We never touch letter codes:
  RAS is built from `increases_toward`.

## 3. Mapping (the core, pure Python, unit-tested outside Slicer)

### 3.1 Geometry: v1 space → IJKToRAS

1. Pick the space: the first space whose axes all have medical `increases_toward` words (left/right/
   anterior/posterior/superior/inferior); else the first space; else none (index space).
2. `to_ras` (3×3, one column per space axis): `left → (-1,0,0)`, `right → (1,0,0)`, `posterior → (0,-1,0)`,
   `anterior → (0,1,0)`, `inferior → (0,0,-1)`, `superior → (0,0,1)`. Must be a permutation with signs (three
   distinct body axes), else reject the space.
3. Unit: scale to millimetres from `unit`/`unit_code` (µm, cm, m, …); unknown units → keep the numbers and
   report.
4. `from_array` (matrix or scale/translation) columns are named array axes; reorder them to I, J, K = last,
   middle, first spatial array axis. `IJKToRAS = [[to_ras · mm · M_cols], [0 0 0 1]]`.
5. Space without directions (non-medical, e.g. `stage`, or `world` written by napari-mlarray): its 1st/2nd/3rd
   axis go to L/P/S (as if it were the DICOM patient frame, which is also what export writes), the numbers are
   kept, and the load reports that the orientation is not anatomical. (Decision to confirm, see §8.)
6. No space at all: unit spacing, identity direction, report.
7. 2-D images: K of size 1, third column = unit normal of the first two (or the space's third axis when the space
   has three axes and the array two), spacing 1. 1-D or >3 space axes: reject with a clear message.

Inverse for export: IJKToRAS → patient space `x/y/z` with `left/posterior/superior` (LPS, millimetre), array axes
`z, y, x` (= K, J, I), `from_array.matrix` = `diag(-1,-1,1) · IJKToRAS` with columns reordered to `[K, J, I]`.
This is exactly MedVol's default, so `.mla` files written by Slicer and by MedVol from the same NIfTI agree.

### 3.2 Non-space axes

| v1 layout | Slicer node |
|---|---|
| 3 space axes (2-D: 2) | scalar volume (or segmentation, §3.3) |
| + one `channel`/`component` axis, ≤ 4 entries, `component` or RGB(A) labels | vector volume (`vtkMRMLVectorVolumeNode`; RGB shown as colour) |
| + one `channel`/`other` axis otherwise (MR echoes, modalities) | one scalar volume per index, named by `axes[].labels` (like napari-mlarray's channel split) |
| + one `time` axis | sequence (`vtkMRMLSequenceNode` + browser) with the time axis' `start/step/unit`; stretch goal, else per-frame volumes |
| anything else (2+ extra axes, no space axes in 3-D) | rejected with the reason |

### 3.3 Values

- `measurement`: physical value = `scale · stored + offset`. If scale/offset are not 1/0, the volume holds the
  physical values (float32, or the smallest integer type that represents them exactly, e.g. CT int16 HU); the
  original dtype/scale/offset are kept for export (§3.5). `unit` → node attribute and window/level preset hint
  (HU → CT preset).
- `label` (index layout): **segmentation node** with one segment per label value (name, colour, terminology
  from `code` when it is an SCT/DCM code), background `0` omitted. Option (module settings): import as labelmap
  volume instead. Values present in the array but missing from the table get a default segment `label <n>`.
- `label` with `label_layout: channel`, `probability`, `instance`: probability → scalar volumes; channel layout
  → one segment per channel (threshold > 0.5 for probabilities is **not** done; binary only); instance →
  labelmap volume (too many segments otherwise). Keep it simple; report what was done.
- `fill`: set as the volume's "no data" attribute only (Slicer has no concept); not applied.

### 3.4 Boxes (`profiles.ml.boxes`)

- `frame: index`: the box `[min, max]` (inclusive voxel centres) becomes a **Markups ROI** aligned with the
  voxel grid: centre = IJKToRAS · centre index, size = (max − min + 1) × spacing, ROI axes = the normalised
  IJK columns (handles flipped and oblique grids exactly; Slicer ROIs can be rotated). Boxes over fewer axes
  than the grid (2-D boxes) span one voxel in the missing axis.
- `frame: <space>`: corners mapped through `to_ras`; axis-aligned ROI in RAS.
- Name `<class name> <score>` (class name from the file's `values.labels` if present), colour from the label
  table or a palette, `label`/`score` and any extra item fields kept as node attributes for export. Boxes are
  grouped under a subject-hierarchy folder `<file> boxes`. Not interactive by default (handles off), fill
  opacity low, outline in slice views (Slicer draws the ROI's intersection in every slice view and the box in
  3-D: this is what napari-bboxes had to build by hand).
- A file with an empty array (`shape` all zero) and boxes = box-only file: only ROIs, geometry from its space.

### 3.5 Export (File → Save, writer confidence)

| node | written as |
|---|---|
| scalar volume | `values.kind: measurement` |
| labelmap volume | `values.kind: label`, label table from the colour node (names, colours) |
| segmentation | rasterised to a labelmap on its reference geometry (overlaps: error with a hint, or `label_layout: channel`?, §8); label table from segments (names, colours, terminology → `code`) |
| vector volume | `component` axis last |
| ROI node(s) | box-only file or appended to a volume's file? (§8) |

- Node loaded from `.mla`: the original metadata is kept (axes order, space name/reference, `source`, `user`,
  profiles, history; the full metadata dict is cached in the scene by node ID and re-read from the original file
  if needed), only geometry/shape/dtype/values are replaced, the array order of the file is restored (it was
  never changed) and a `history` entry `{action: edited, tool: SlicerMLArray x.y}` is added. Geometry that did not
  change (IJKToRAS equal within 1e-6) keeps its exact stored matrix (no float drift).
- New node: patient space LPS as §3.1, `writer`/`history` set, Slicer's node name in `user`? (no: nothing
  invented). `source` empty.
- Physical values written back with the original scale/offset when they still fit; else stored as is.

## 4. Package layout

```
SlicerMLArray/                    Slicer extension (CMake, for the Extensions Index)
  CMakeLists.txt                  extension: name, homepage, category "Informatics", icon, screenshot URLs
  SlicerMLArray/                  scripted module
    CMakeLists.txt
    SlicerMLArray.py              hidden module + FileReader/FileWriter + Settings panel (labels as seg/labelmap,
                                  boxes on/off), self-test (ScriptedLoadableModuleTest)
    SlicerMLArrayLib/             pure Python, no Slicer imports
      geometry.py                 §3.1 (space ↔ IJKToRAS)
      layout.py                   §3.2 (what node(s) a file becomes)
      values.py                   §3.3 (scaling, label table ↔ segments, terminology)
      boxes.py                    §3.4 (box ↔ ROI centre/size/axes)
      export.py                   §3.5 (metadata for a node)
      dependencies.py             import / pip_install("mlarray>=1,<2"), version check
    SlicerMLArrayLib/slicer_io.py Slicer bridge (nodes, scene, subject hierarchy)
    Resources/Icons/SlicerMLArray.png
tests/                            pytest for SlicerMLArrayLib (no Slicer)
scripts/verify_in_slicer.py       runs in Slicer (headless + on screen), §6
docs/                             README, this plan
```

The pyproject stays only to run the pure-Python tests (no PyPI release; the current workflow publishes to PyPI
with a token secret and was never used: remove that job). Distribution: Slicer Extensions Index (a
`SlicerMLArray.s4ext`/JSON PR to `Slicer/ExtensionsIndex`), plus "add module path" for development.

## 5. Stages

| # | step | done |
|---|---|---|
| 1 | Install Slicer 5.12.4 (latest stable, `~/Slicer-5.12.4-linux-amd64`), remove 5.10.0; re-run KB `slicer_geometry.py` facts on 5.12.4 (identical to 5.10.0) | ☑ |
| 2 | Delete the v0 code and tests; new layout (§4); CI runs pytest on 3.10/3.12/3.13 with mlarray 1.0 | ☑ `a505153` |
| 3 | `geometry.py` + tests: IJKToRAS equals the NIfTI affine via MedVol for ras/lps/sagittal/mirrored/oblique, NRRD, axis orders, units, 2-D, round trips | ☑ `a505153` (36 tests) |
| 4 | Reader for scalar/vector/split channels/sequences + values scaling | ☑ `a505153` |
| 5 | Labels → segmentation (names, colours, terminology) and labelmap option | ☑ `a505153` |
| 6 | Boxes → ROIs (index and space frames, box-only files) | ☑ `a505153` |
| 7 | Writer: volumes, labelmaps, segmentations; ROIs via the module panel; keep metadata of loaded files | ☑ `a505153` |
| 8 | Verification in Slicer 5.12.4: `scripts/verify_in_slicer.py` 95/95 (7 KB phantom grids vs Slicer's readers, layouts, round trips, 25 example cases), on-screen screenshots (letters read radiologically, boxes on organs), ioManager open/save, first-use install, module panel | ☑ `9995bf0` |
| 9 | README, Extensions Index entry (`docs/extensions-index/`), screenshot; merge into `main` and push after approval | README ☑ `10dd8ee`; merge/push open |
| 10 | Submit to the Extensions Index (user decision) | |

## 6. Verification

- **Pure Python** (pytest, `medvol-v2` env): geometry against MedVol (`medvol.read` of NIfTI phantoms vs
  `geometry.ijk_to_ras` of the converted `.mla`), round trips, layouts, label tables, boxes.
- **In Slicer** (headless `Slicer --no-main-window --python-script`, in docker `orient-viewers` with Xvfb like the
  KB experiments `experiments/verify_slicer_mitk/`):
  1. KB letter phantoms (NIfTI ras/lps/sag/obl25, NRRD): load the NIfTI with Slicer's own reader and the
     MedVol-converted `.mla` with SlicerMLArray; IJKToRAS and arrays must be identical (Slicer's NIfTI reader is
     the reference; KB verified it).
  2. Screenshots of the three slice views + 3-D: the phantom letters read correctly (same check as the KB).
  3. Example data `/home/k539i/data/mlarray/tmp/*`: image + seg + boxes load, every box encloses its segment
     component (RAS bounds of segment ⊂ ROI bounds), names/scores shown.
  4. Save → reload round trips (volume, labelmap, segmentation, ROIs), and the saved `.mla` opened in
     napari-mlarray lines up with the original (same world positions).
  5. Dependency path: a clean Slicer (fresh settings dir) installs `mlarray` on first load after the prompt.

## 7. Out of scope

- Reading other formats through MedVol (Slicer reads NIfTI/NRRD/DICOM itself).
- DICOM SEG/RT export from Slicer to `.mla` beyond the label table (`source` is not synthesised).
- Pyramids, non-linear `profile_mapping`.

## 8. Decisions (confirmed by the user 2026-11-06: all recommendations; submit to the Extensions Index after their approval)

1. Labels default: segmentation node (recommended; names/colours/codes visible in Segment Editor) vs labelmap.
2. Spaces without anatomical directions: show on L/P/S axes with a warning (recommended) or reject.
3. Segmentation export with overlapping segments: error with a hint (recommended) or `label_layout: channel`.
4. Saving ROIs: separate box-only `.mla` (recommended, like the example `_boxes.mla`) vs into the image file.
5. Distribution: Extensions Index submission at the end, or only "add module path" for now.

## 9. Deviations from the plan and findings

- **ROI export** is not in File > Save (Slicer's save dialog writes one file per node, which does not fit one
  box file per image): the module panel *Export boxes* (and `logic.saveBoxes`) writes selected ROIs to a box-only
  file, in a reference image's indices, in the ROIs' original file frame, or in patient LPS coordinates.
- **Box axes the box does not name** span the whole image (not one voxel).
- **`label_layout: channel`:** channel 0 is the background unless the label table names it; it is not loaded as
  a segment and is rebuilt on save as "in no other channel".
- **Slicer's own NIfTI reader reverses K for a left-handed grid** (sagittal phantom stored with a flipped first
  axis): IJKToRAS and the array differ from nibabel/SimpleITK/MedVol, but every voxel is at the same world position.
  SlicerMLArray keeps the stored array order (matches nibabel/SimpleITK). The verification therefore compares world
  positions through the voxel index mapping, not raw matrices.
- **GitHub:** there is no public GitHub repository yet (`MIC-DKFZ/SlicerMLArray` does not exist; the GitLab remote
  is `hi-dkfz/applied-computer-vision-lab/tools/slicermlarray`). The Extensions Index needs a public repository URL
  for `scm_url`/homepage: create the GitHub mirror before submitting.

## 10. Extensions Index submission (after the user's approval)

Checklist (https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#submit-the-extension-to-the-extensions-index):
1. Public GitHub repository `MIC-DKFZ/SlicerMLArray` (mirror of the GitLab repo), `main` = `rework/v1`; the README
   screenshot and `SlicerMLArray.png` must be reachable at the URLs in `CMakeLists.txt`.
2. Repository topic `3d-slicer-extension`; license (MIT), README with screenshot: done.
3. Fork `Slicer/ExtensionsIndex`, add `docs/extensions-index/SlicerMLArray.json` as `SlicerMLArray.json` on `main`
   (preview builds), open a PR; the PR template's checklist is answered by this repo. After the nightly build is
   green, backport to the `5.12` branch so it appears in the stable Extensions Manager.
4. Not done locally: a CMake build against a Slicer build tree (the extension is pure Python; the factory builds it).

SlicerMLArray
=============

Python-based 3D Slicer integration for loading and saving MLArray (`.mla`) image volumes through Slicer's normal file dialogs.

## What it does

- loads supported `.mla` files as Slicer volumes
- saves scalar and labelmap Slicer volumes back to `.mla`
- preserves spacing, origin, direction, dtype, and selected metadata
- installs the `mlarray` Python dependency inside Slicer on demand when missing

## Supported layouts

Load:

- 3D scalar MLArray images
- 3D labelmap MLArray images when `meta.is_seg=True`
- 4D vector MLArray images only when axis labels explicitly identify one `channel` or `components` axis and explicit `spatial_z`, `spatial_y`, `spatial_x`

Save:

- `vtkMRMLScalarVolumeNode`
- `vtkMRMLLabelMapVolumeNode`

Unsupported layouts fail explicitly. This MVP does not reinterpret ambiguous tensors, temporal stacks, or unlabeled 4D arrays.

## Repository layout

- `SlicerMLArray.py`: hidden scripted Slicer module and file reader/writer entry points
- `slicermlarray/conversion.py`: pure-Python axis and geometry mapping logic
- `slicermlarray/dependencies.py`: runtime dependency checks and installation flow
- `slicermlarray/slicer_io.py`: Slicer runtime bridge
- `DESIGN.md`: design note based on the inspected local MLArray and Slicer codebases

## Development

Install the package for local testing:

```bash
pip install -e .[dev]
pytest
```

If you also want real `mlarray` runtime tests outside Slicer, install `mlarray` into your Python environment as well.

## Trying it in the local Slicer installation

Target Slicer install:

- `/home/k539i/Slicer-5.10.0-linux-amd64`

Steps:

1. Start Slicer from `/home/k539i/Slicer-5.10.0-linux-amd64/Slicer`.
2. Open `Edit -> Application Settings -> Modules`.
3. Add `/home/k539i/Documents/projects/SlicerMLArray` to `Additional module paths`.
4. Restart Slicer.
5. Open a `.mla` file using `File -> Add Data` or `File -> Open`.
6. Save a supported loaded or existing scalar/labelmap volume as `.mla` through `File -> Save`.

On the first `.mla` load/save in a Slicer Python environment that does not already contain `mlarray`, the extension prompts to install it via `slicer.util.pip_install("mlarray")`.

## Mapping policy

- Slicer numpy arrays are treated as `(z, y, x)`.
- Saved MLArray files use explicit axis labels `["spatial_z", "spatial_y", "spatial_x"]`.
- Geometry is permuted between MLArray spatial-axis order and Slicer's `(x, y, z)` IJK/RAS geometry representation.
- Labelmap export sets `meta.is_seg=True`.
- Namespaced metadata is written under `extra.slicer_mlarray`.

## Known limitations

- no custom MRML storage node yet, so default scene-storage behavior is not fully `.mla`-native
- no vector-volume export in this MVP
- no temporal/ND tensor support
- external `.mla` files with spatial metadata in a non-RAS world convention are not auto-detected

# SlicerMLArray Design

## Verified MLArray findings

Inspected local source: `/home/k539i/Documents/projects/mlarray`

- Primary API is `mlarray.MLArray`.
- Load paths:
  - `MLArray(path)` loads an `.mla` file eagerly enough for normal use.
  - `MLArray.open(path, mmap_mode="r")` opens lazily through Blosc2 and still exposes metadata immediately.
- Save path:
  - `MLArray(array, spacing=..., origin=..., direction=..., axis_labels=..., meta=...).save(path)`.
- Metadata model:
  - Spatial metadata lives under `meta.spatial`.
  - Standard fields are `spacing`, `origin`, `direction`, `affine`, `shape`, `axis_labels`, `axis_units`.
  - Segmentation intent is represented by `meta.is_seg`.
  - Free-form JSON-serializable metadata lives in `meta.source` and `meta.extra`.
- Array behavior:
  - `shape`, `dtype`, `ndim`, `spatial_ndim`, slicing, and `to_numpy()` are provided directly on `MLArray`.
  - `.mla` storage is Blosc2-backed and supports lazy/memory-mapped access through `MLArray.open(...)`.
- Axis semantics:
  - `axis_labels` are optional.
  - Explicit labels include `spatial_x`, `spatial_y`, `spatial_z`, `channel`, `components`, and others.
  - MLArray examples/documentation state that spatial metadata ordering follows the spatial axis order, not an external fixed x/y/z convention.
- File format behavior:
  - `.mla` requires the `.mla` suffix.
  - Data and metadata are stored through Blosc2, with extra metadata in variable-length metadata.

## Verified Slicer findings

Inspected local installation: `/home/k539i/Slicer-5.10.0-linux-amd64`

- Scripted file readers/writers are discovered from specially named classes in a scripted module:
  - `<ModuleName>FileReader`
  - `<ModuleName>FileWriter`
- Relevant reference code:
  - `/home/k539i/Slicer-5.10.0-linux-amd64/lib/Slicer-5.10/qt-scripted-modules/SlicerScriptedFileReaderWriterTest.py`
  - `/home/k539i/Slicer-5.10.0-linux-amd64/lib/Slicer-5.10/qt-scripted-modules/ImportItkSnapLabel.py`
- Reader API:
  - `description()`
  - `fileType()`
  - `extensions()`
  - `canLoadFileConfidence(filePath)` or `canLoadFile(filePath)`
  - `load(properties)`
- Writer API:
  - `description()`
  - `fileType()`
  - `extensions(obj)`
  - `canWriteObjectConfidence(obj)`
  - `write(properties)`
- Loaded/written node IDs are reported through `self.parent.loadedNodes` and `self.parent.writtenNodes`.
- Volume/numpy conversion:
  - `slicer.util.arrayFromVolume(node)` returns scalar data as `(z, y, x)` and vector data as `(z, y, x, c)`.
  - `slicer.util.addVolumeFromArray(array, ijkToRAS=..., nodeClassName=...)` creates scalar, labelmap, or vector volumes from numpy arrays.
- Geometry representation:
  - `GetSpacing()`, `GetOrigin()`, and `GetIJKToRASDirectionMatrix(...)` expose volume geometry.
  - `SetIJKToRASMatrix(...)` applies full geometry to a node.
  - Slicer’s IJK columns are ordered `(x, y, z)`, while numpy volume arrays are ordered `(z, y, x)`.
- Dependency installation:
  - `slicer.util.pip_install(...)` is the supported Python-package installer inside Slicer.

## Chosen mapping rules

- Supported MLArray import layouts:
  - 3D unlabeled arrays: assumed `(z, y, x)` scalar images.
  - 3D arrays with explicit `spatial_z`, `spatial_y`, `spatial_x` in any axis order: imported as scalar or labelmap.
  - 4D arrays with exactly one `channel` or `components` axis plus explicit `spatial_z`, `spatial_y`, `spatial_x`: imported as vector volumes.
- Unsupported import layouts:
  - Generic `spatial` labels.
  - Any temporal or other non-image axes.
  - Unlabeled 4D arrays.
  - Segmentation-like arrays with a channel/components axis.
- Slicer export scope:
  - `vtkMRMLScalarVolumeNode` -> `.mla`
  - `vtkMRMLLabelMapVolumeNode` -> `.mla` with `meta.is_seg=True`
  - Vector export is intentionally left out of the MVP.

## Geometry correctness

- Slicer arrays are handled in numpy `(z, y, x)` order.
- Slicer geometry uses IJK/RAS `(x, y, z)` order.
- MLArray metadata ordering is treated as following MLArray spatial axis order.
- Therefore:
  - On import, MLArray spatial metadata is permuted into Slicer `(x, y, z)` before building the IJK-to-RAS matrix.
  - On export, Slicer `(x, y, z)` geometry is permuted back into MLArray `(z, y, x)` metadata order.
- This preserves Slicer round-trips and matches the local MLArray examples that describe metadata as following the spatial axis order.

## Important assumption

MLArray itself does not define a Slicer-specific world-coordinate convention such as RAS vs LPS. For this extension:

- imported MLArray spatial metadata is interpreted directly as Slicer RAS-space metadata after the axis-order permutation described above
- exported files record a namespaced `extra.slicer_mlarray.coordinate_system = "RAS"` hint

This is correct for files written by this plugin and for external `.mla` files that already store spatial metadata in the same convention.

## Native integration limitations

- The scripted reader/writer integrates with Slicer’s normal Open/Save UI.
- It does not provide a custom MRML storage node, so scene-bundle or non-interactive storage-node-driven save paths may still prefer Slicer’s native volume storage formats unless the user explicitly saves as `.mla`.

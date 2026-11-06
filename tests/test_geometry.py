"""IJKToRAS from MLArray spaces, checked against the files' own geometry (nibabel/pynrrd) via MedVol."""

import warnings

import nibabel as nib
import nrrd
import numpy as np
import pytest

import medvol
from mlarray import Meta
from SlicerMLArrayLib.errors import UnsupportedFile
from SlicerMLArrayLib.geometry import grid, patient_space, updated_space

SHAPE = (6, 7, 8)          # file i, j, k
AFFINES = {                # NIfTI (RAS), columns i, j, k
    "ras": [[1.0, 0, 0, -10], [0, 1.25, 0, 20], [0, 0, 2.0, -30], [0, 0, 0, 1]],
    "lps": [[-1.0, 0, 0, 10], [0, -1.25, 0, 20], [0, 0, 2.0, -30], [0, 0, 0, 1]],
    "sagittal": [[0, 0, -1.5, 40], [1.25, 0, 0, -20], [0, -2.0, 0, 30], [0, 0, 0, 1]],
    "mirrored": [[-1.0, 0, 0, 10], [0, 1.25, 0, 20], [0, 0, 2.0, -30], [0, 0, 0, 1]],
}


def _oblique():
    angle = np.deg2rad(25)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    out = np.eye(4)
    out[:3, :3] = rotation @ np.diag([0.9, 1.1, 2.5])
    out[:3, 3] = [5, -6, 7]
    return out.tolist()


AFFINES["oblique"] = _oblique()


def _read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image, _ = medvol.read(path)
    return image


@pytest.mark.parametrize("name", sorted(AFFINES))
def test_nifti_ijk_to_ras_is_the_nifti_affine(tmp_path, name):
    """MedVol keeps the file order (I = file i is the last array axis), so IJKToRAS must be the NIfTI affine."""
    affine = np.array(AFFINES[name])
    img = nib.Nifti1Image(np.zeros(SHAPE, np.int16), affine)
    img.set_qform(affine, 1)
    img.set_sform(affine, 1)
    nib.save(img, tmp_path / "a.nii.gz")
    image = _read(tmp_path / "a.nii.gz")
    g = grid(image.meta)
    np.testing.assert_allclose(g.ijk_to_ras, affine, atol=1e-5)
    assert g.anatomical and not g.notes
    # the same after a round trip through an .mla file
    medvol.write(image, tmp_path / "a.mla")
    from mlarray import MLArray

    np.testing.assert_allclose(grid(MLArray.open(tmp_path / "a.mla", mmap_mode="r").meta).ijk_to_ras, affine, atol=1e-5)


def test_nrrd_lps_directions(tmp_path):
    directions = np.array([[0, 1.25, 0], [0, 0, -2.0], [1.5, 0, 0]])       # rows: file axes i, j, k (LPS)
    origin = np.array([5.0, -7.0, 9.0])
    nrrd.write(str(tmp_path / "a.nrrd"), np.zeros(SHAPE, np.int16),
               {"space": "left-posterior-superior", "space directions": directions, "space origin": origin})
    expected = np.eye(4)
    expected[:3, :3] = np.diag([-1.0, -1.0, 1.0]) @ directions.T
    expected[:3, 3] = np.diag([-1.0, -1.0, 1.0]) @ origin
    np.testing.assert_allclose(grid(_read(tmp_path / "a.nrrd").meta).ijk_to_ras, expected, atol=1e-6)


def _meta(axes, spaces, shape):
    return Meta(axes=axes, spaces=spaces, shape=shape)


LPS_AXES = [{"name": "x", "unit": "millimeter", "increases_toward": "left"},
            {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
            {"name": "z", "unit": "millimeter", "increases_toward": "superior"}]


def test_any_axis_order_and_direction_words():
    # array (x, z, y) in a space whose axes are (superior, right, anterior)
    space_axes = [{"name": "s", "unit": "mm", "increases_toward": "superior"},
                  {"name": "r", "unit": "mm", "increases_toward": "right"},
                  {"name": "a", "unit": "mm", "increases_toward": "anterior"}]
    matrix = [[0, 2.0, 0, 1], [3.0, 0, 0, 2], [0, 0, 4.0, 3], [0, 0, 0, 1]]     # columns: z, x, y
    meta = _meta(["x", "z", "y"], [{"name": "p", "axes": space_axes,
                                    "from_array": {"array_axes": ["z", "x", "y"], "matrix": matrix}}], (2, 3, 4))
    g = grid(meta)
    assert g.spatial_axes == (0, 1, 2)
    # s = 2 x, r = 3 z, a = 4 y; I = y -> anterior 4, J = z -> right 3, K = x -> superior 2
    expected = np.array([[0, 3.0, 0, 2], [4.0, 0, 0, 3], [0, 0, 2.0, 1], [0, 0, 0, 1]])
    np.testing.assert_allclose(g.ijk_to_ras, expected)


def test_units_are_converted_to_millimetre():
    axes = [dict(a, unit="micrometer") for a in LPS_AXES]
    meta = _meta(["z", "y", "x"], [{"name": "p", "axes": axes,
                                    "from_array": {"array_axes": ["x", "y", "z"], "scale": [2.0, 2.0, 500.0],
                                                   "translation": [0.0, 0.0, 0.0]}}], (2, 3, 4))
    np.testing.assert_allclose(np.abs(np.diag(grid(meta).ijk_to_ras)[:3]), [0.002, 0.002, 0.5])


def test_space_without_directions_is_shown_as_lps_with_a_note():
    axes = [{"name": "z", "unit": "micrometer"}, {"name": "y", "unit": "micrometer"}, {"name": "x", "unit": "micrometer"}]
    meta = _meta(["z", "y", "x"], [{"name": "stage", "axes": axes,
                                    "from_array": {"array_axes": ["z", "y", "x"], "scale": [1000.0, 1000.0, 1000.0],
                                                   "translation": [0.0, 0.0, 0.0]}}], (2, 3, 4))
    g = grid(meta)
    assert not g.anatomical and any("no anatomical directions" in n for n in g.notes)
    # stage axis 1 (z) -> left, 2 (y) -> posterior, 3 (x) -> superior; I = x
    np.testing.assert_allclose(g.ijk_to_ras[:3, :3], [[0, 0, -1.0], [0, -1.0, 0], [1.0, 0, 0]])


def test_no_space_is_index_space():
    g = grid(Meta(axes=["z", "y", "x"], shape=(2, 3, 4)))
    np.testing.assert_allclose(g.ijk_to_ras, np.diag([-1.0, -1.0, 1.0, 1.0]))
    assert g.notes


def test_2d_image_in_3d_space_gets_a_right_handed_normal():
    # a coronal 2-D image: rows go inferior, columns go left
    matrix = [[0, 0.5, 1], [0, 0, 2], [-0.5, 0, 3], [0, 0, 1]]
    meta = _meta(["y", "x"], [{"name": "patient", "axes": LPS_AXES,
                               "from_array": {"array_axes": ["y", "x"], "matrix": matrix}}], (3, 4))
    g = grid(meta)
    assert g.is_2d and g.spatial_axes == (0, 1)
    i, j, k = g.ijk_to_ras[:3, :3].T
    np.testing.assert_allclose(i, [-0.5, 0, 0])
    np.testing.assert_allclose(j, [0, 0, -0.5])
    np.testing.assert_allclose(k, np.cross(i, j) / np.linalg.norm(np.cross(i, j)))


def test_too_many_space_axes():
    with pytest.raises(UnsupportedFile):
        grid(Meta(axes=["t", "z", "y", "x"], shape=(2, 2, 2, 2)))


def test_patient_space_round_trip():
    ijk = np.array(_oblique())
    space = patient_space(ijk, ["z", "y", "x"])
    meta = _meta(["z", "y", "x"], [space], (2, 3, 4))
    np.testing.assert_allclose(grid(meta).ijk_to_ras, ijk, atol=1e-12)


def test_updated_space_keeps_its_axes_and_columns():
    space_axes = [{"name": "r", "unit": "mm", "increases_toward": "right"},
                  {"name": "a", "unit": "mm", "increases_toward": "anterior"},
                  {"name": "s", "unit": "mm", "increases_toward": "superior"}]
    meta = _meta(["z", "y", "x"], [{"name": "scanner", "axes": space_axes,
                                    "from_array": {"array_axes": ["x", "y", "z"], "scale": [1.0, 1.0, 1.0],
                                                   "translation": [0.0, 0.0, 0.0]}}], (2, 3, 4))
    new_ijk = np.array(_oblique())
    out = updated_space(meta.spaces[0], new_ijk, ["z", "y", "x"])
    assert out["name"] == "scanner" and out["from_array"]["array_axes"] == ["x", "y", "z"]
    np.testing.assert_allclose(grid(_meta(["z", "y", "x"], [out], (2, 3, 4))).ijk_to_ras, new_ijk, atol=1e-12)

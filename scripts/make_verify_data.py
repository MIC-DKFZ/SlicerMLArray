"""Test data for ``verify_in_slicer.py`` (run with a Python that has mlarray 1.x, MedVol and nibabel).

    python make_verify_data.py PHANTOM_DIR OUT_DIR

PHANTOM_DIR holds the orientation-KB letter phantom (``image-orientation-kb/experiments/verify_slicer_mitk/
make_phantom.py``): the same world content on RAS, LPS and oblique grids as NIfTI and NRRD. Each is converted to
``.mla`` with MedVol, and a sagittal-stored, mirrored copy is added. Slicer must place the ``.mla`` exactly where
its own NIfTI/NRRD reader places the original.

Also written: small files for every layout (2-D, RGB, channels, time, scaled CT, label channels, codes).
"""

import shutil
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

import medvol
from mlarray import MLArray

PHANTOMS = ["ras.nii.gz", "lps.nii.gz", "lps.nrrd", "oblique.nii.gz", "oblique.nrrd", "lps_space_RAS.nrrd"]
LPS_AXES = [{"name": "x", "unit": "millimeter", "increases_toward": "left"},
            {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
            {"name": "z", "unit": "millimeter", "increases_toward": "superior"}]


def patient(array_axes, matrix):
    return [{"name": "patient", "axes": LPS_AXES, "from_array": {"array_axes": array_axes, "matrix": matrix}}]


def main(phantoms: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "phantom").mkdir(exist_ok=True)
    names = list(PHANTOMS)
    # sagittal-stored and mirrored: file axes (k, i, j) of the RAS phantom, first one flipped
    ras = nib.load(phantoms / "ras.nii.gz")
    data = np.asarray(ras.dataobj).transpose(2, 0, 1)[::-1]
    affine = ras.affine[:, [2, 0, 1, 3]].copy()
    affine[:3, 3] += affine[:3, 0] * (data.shape[0] - 1)
    affine[:3, 0] *= -1
    sagittal = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    sagittal.set_qform(affine, 1)
    sagittal.set_sform(affine, 1)
    nib.save(sagittal, out / "phantom" / "sagittal.nii.gz")
    names.append("sagittal.nii.gz")
    for name in PHANTOMS:
        shutil.copy(phantoms / name, out / "phantom" / name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            stem = name.replace(".nii.gz", "").replace(".nrrd", "") + ("_nrrd" if name.endswith(".nrrd") else "")
            medvol.convert(out / "phantom" / name, out / "phantom" / f"{stem}.mla")

    layouts = out / "layouts"
    layouts.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    m3 = [[0, 0, 0.8, -10], [0, 0.7, 0, 20], [2.5, 0, 0, -30], [0, 0, 0, 1]]          # columns z, y, x
    # 2-D coronal image: rows inferior, columns left
    MLArray(rng.integers(0, 100, (30, 40)).astype(np.int16), axes=["y", "x"],
            spaces=patient(["y", "x"], [[0, 0.5, 1], [0, 0, 2], [-0.5, 0, 3], [0, 0, 1]])).save(layouts / "coronal_2d.mla")
    rgb = rng.integers(0, 255, (8, 10, 12, 3)).astype(np.uint8)
    MLArray(rgb, axes=["z", "y", "x", {"name": "c", "type": "channel", "labels": ["red", "green", "blue"]}],
            spaces=patient(["z", "y", "x"], m3)).save(layouts / "rgb.mla")
    MLArray(rng.normal(size=(2, 8, 10, 12)).astype(np.float32),
            axes=[{"name": "c", "type": "channel", "labels": ["T1", "T2"]}, "z", "y", "x"],
            spaces=patient(["z", "y", "x"], m3)).save(layouts / "channels.mla")
    MLArray(rng.normal(size=(3, 8, 10, 12)).astype(np.float32),
            axes=[{"name": "t", "type": "time", "start": 0.0, "step": 2.0, "unit": "second"}, "z", "y", "x"],
            spaces=patient(["z", "y", "x"], m3)).save(layouts / "time.mla")
    MLArray(rng.integers(0, 4096, (8, 10, 12)).astype(np.uint16), axes=["z", "y", "x"],
            spaces=patient(["z", "y", "x"], m3),
            values={"kind": "measurement", "unit": "hounsfield_unit", "scale": 1.0, "offset": -1024.0}).save(layouts / "ct_scaled.mla")
    onehot = np.zeros((3, 8, 10, 12), np.uint8)
    onehot[1, 2:5, 2:5, 2:5] = 1
    onehot[2, 3:7, 4:8, 4:9] = 1                              # overlaps channel 1: allowed in channel layout
    onehot[0] = onehot[1:].max(axis=0) == 0                  # background channel
    MLArray(onehot, axes=[{"name": "c", "type": "channel"}, "z", "y", "x"], spaces=patient(["z", "y", "x"], m3),
            values={"kind": "label", "label_layout": "channel",
                    "labels": {"1": {"name": "kidney", "color": [200, 50, 50]}, "2": {"name": "cyst"}}}
            ).save(layouts / "label_channels.mla")
    seg = np.zeros((8, 10, 12), np.uint8)
    seg[2:5, 2:5, 2:5] = 1
    seg[5:7, 6:9, 7:11] = 3
    MLArray(seg, axes=["z", "y", "x"], spaces=patient(["z", "y", "x"], m3),
            values={"kind": "label", "labels": {
                "0": {"name": "background"},
                "1": {"name": "liver", "color": [221, 130, 101],
                      "code": {"scheme": "SCT", "value": "10200004", "meaning": "Liver"}},
                "3": {"name": "spleen", "color": [157, 108, 162]}}},
            profiles={"ml": {"boxes": {"frame": "patient", "axes": ["x", "y", "z"],
                                       "items": [{"min": [-8, 21, -25], "max": [-5, 23, -18], "label": 1, "score": 0.5}]}}},
            ).save(layouts / "seg_codes.mla")
    print(f"written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

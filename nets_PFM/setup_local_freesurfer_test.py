#!/usr/bin/env python3
"""
One-off: build a self-consistent native-surface set for a single ds-asd subject
from a LOCAL recon-all run (FreeSurfer 7.3.2), for smoke-testing the PFM
pipeline while the cluster (source of the real FreeSurfer 6.0.1 dirs) is down
for maintenance.

Why this script exists: 02_surface_cifti.py needs smoothwm/pial/midthickness
(in T1w space) plus sphere.reg, and these must all come from the SAME
recon-all run to have matching vertex topology. fMRIPrep's already-exported
anat/*.surf.gii for this subject came from a different (cluster, FreeSurfer
6.0.1) recon-all run, so they can't be mixed with a freshly-run sphere.reg.
This script regenerates smoothwm/pial/midthickness from the local recon-all
output instead, replicating fMRIPrep's own surface-export recipe (mris_convert
+ add c_ras + apply the fsnative->T1w ITK affine fMRIPrep already computed).

Usage:
  python setup_local_freesurfer_test.py sub-01

Expects:
  - Local recon-all already completed at config.FREESURFER_OVERRIDE[subject]
    (i.e. <that path>/sub-01/surf/{lh,rh}.{smoothwm,pial,sphere.reg})
  - fMRIPrep's from-fsnative_to-T1w_mode-image_xfm.txt already present at
    FMRIPREP/sub-01/ses-1/anat/ (used unchanged — it's a volume-conformation
    transform, not mesh-topology-dependent, so it's still valid here)

Writes smoothwm/pial/midthickness gifti to config.ANAT_DIR_OVERRIDE[subject].
sphere.reg conversion is handled separately, automatically, by
02_surface_cifti.py's ensure_sphere_reg_gii() the first time the pipeline runs.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent))
from config import FMRIPREP, SESSION, FREESURFER_OVERRIDE, ANAT_DIR_OVERRIDE, WB_COMMAND


def load_itk_affine(xfm_path: Path) -> np.ndarray:
    """Parse an ANTs/ITK 'Insight Transform File' affine into a 4x4 RAS matrix.

    ITK transforms operate in LPS; nibabel/FreeSurfer surfaces are in RAS.
    Converting requires sandwiching with diag(-1, -1, 1) on both sides.
    """
    text = xfm_path.read_text()
    if 'Insight Transform File' not in text:
        raise ValueError(f'Not an ITK transform file: {xfm_path}')

    params = fixed = None
    for line in text.splitlines():
        if line.startswith('Parameters:'):
            params = np.fromstring(line.split(':', 1)[1], sep=' ')
        elif line.startswith('FixedParameters:'):
            fixed = np.fromstring(line.split(':', 1)[1], sep=' ')
    if params is None or fixed is None:
        raise ValueError(f'Could not parse affine from {xfm_path}')

    matrix = params[:9].reshape(3, 3)
    translation = params[9:12]
    center = fixed[:3]

    # ITK affine (about `center`): y = A @ (x - center) + translation + center
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = matrix
    affine_lps[:3, 3] = translation + center - matrix @ center

    lps2ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps2ras @ affine_lps @ lps2ras


def convert_surface(fs_surf_path: Path, affine_ras: np.ndarray, out_gii: Path) -> None:
    """mris_convert a FreeSurfer surface to gifti, then move tkr-RAS coords
    (relative to the conformed volume centre) into T1w scanner RAS, matching
    fMRIPrep's own surface-export convention."""
    tmp_gii = out_gii.with_suffix('.tmp.surf.gii')
    subprocess.run(['mris_convert', str(fs_surf_path), str(tmp_gii)], check=True)

    img = nib.load(tmp_gii)
    points = img.darrays[0]
    c_ras = np.array([
        float(points.meta['VolGeomC_R']),
        float(points.meta['VolGeomC_A']),
        float(points.meta['VolGeomC_S']),
    ])

    coords = points.data.astype(np.float64) + c_ras            # tkr-RAS -> fsnative scanner RAS
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
    coords_t1w = (affine_ras @ coords_h.T).T[:, :3]              # fsnative scanner RAS -> T1w RAS

    points.data = coords_t1w.astype(np.float32)
    img.to_filename(out_gii)
    tmp_gii.unlink()


def main(subject: str) -> None:
    if subject not in FREESURFER_OVERRIDE or subject not in ANAT_DIR_OVERRIDE:
        raise SystemExit(
            f'{subject} has no FREESURFER_OVERRIDE/ANAT_DIR_OVERRIDE entry in '
            f'config_asd.py — add one before running this script.'
        )

    fs_subjects_dir = FREESURFER_OVERRIDE[subject]
    fs_surf_dir = fs_subjects_dir / subject / 'surf'
    out_dir = ANAT_DIR_OVERRIDE[subject]
    out_dir.mkdir(parents=True, exist_ok=True)

    xfm_path = FMRIPREP / subject / SESSION / 'anat' / f'{subject}_{SESSION}_from-fsnative_to-T1w_mode-image_xfm.txt'
    affine_ras = load_itk_affine(xfm_path)

    for hemi, hemi_fs in [('L', 'lh'), ('R', 'rh')]:
        smoothwm_out = out_dir / f'{subject}_{SESSION}_hemi-{hemi}_smoothwm.surf.gii'
        pial_out     = out_dir / f'{subject}_{SESSION}_hemi-{hemi}_pial.surf.gii'
        midthick_out = out_dir / f'{subject}_{SESSION}_hemi-{hemi}_midthickness.surf.gii'

        convert_surface(fs_surf_dir / f'{hemi_fs}.smoothwm', affine_ras, smoothwm_out)
        convert_surface(fs_surf_dir / f'{hemi_fs}.pial', affine_ras, pial_out)

        subprocess.run([
            WB_COMMAND, '-surface-average', str(midthick_out),
            '-surf', str(smoothwm_out), '-surf', str(pial_out),
        ], check=True)
        print(f'[{hemi}] wrote smoothwm, pial, midthickness -> {out_dir}')

    print(f'\nDone. sphere.reg will be auto-converted on first pipeline run '
          f'from {fs_surf_dir}.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(f'Usage: {sys.argv[0]} sub-XX')
    main(sys.argv[1])

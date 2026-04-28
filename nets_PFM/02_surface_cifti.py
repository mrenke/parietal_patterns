#!/usr/bin/env python3
"""
Step 2 — Surface mapping + CIFTI generation (Gordon 2017 protocol).

Per run:
  1. Ribbon-constrained volume → fsnative surface (wb_command)
  2. Resample fsnative → fsLR 32k (wb_command)
  3. Extract subcortical timeseries from T1w volume + create label volume
  4. Assemble CIFTI dtseries (wb_command)
  5. Smooth CIFTI, σ = 2.55 mm (wb_command)

  output: 
  sub-XX_ses-1_task-magjudge_run-X_space-fsLR32k_bold_smooth.dtseries.nii
  -->  CIFTI, surface + subcortical combined, NIfTI-2 container (intent code in the header marks it as a CIFTI dtseries)

Then concatenate all runs into one session CIFTI.

Usage:
  python 02_surface_cifti.py sub-01
"""
import sys
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (FMRIPREP, FREESURFER, OUTPUT_ROOT, SESSION, TASK, RUNS,
                    TR, SMOOTH_SIGMA, WB_COMMAND, FSLR_SPHERE, FSLR_MIDTHICK,
                    FSLR_ROI, SUBCORTICAL_LABELS)


# ---------------------------------------------------------------------------
# wb_command wrapper
# ---------------------------------------------------------------------------

def wb(args: list, check: bool = True) -> None:
    cmd = [WB_COMMAND] + [str(a) for a in args]
    print('    wb_command ' + ' '.join(str(a) for a in args[:4]) + ' ...')
    subprocess.run(cmd, check=check)


# ---------------------------------------------------------------------------
# Subcortical label volume helpers
# ---------------------------------------------------------------------------

def make_label_volume(aparcaseg_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """
    Create:
      - A remapped label NIFTI (aparc+aseg values → CIFTI integer IDs)
      - A label list .txt file for wb_command -volume-label-import
    Returns (label_vol_path, label_list_path).
    """
    seg_img  = nib.load(aparcaseg_path)
    seg_data = np.asarray(seg_img.dataobj, dtype=np.int32)

    remapped = np.zeros_like(seg_data)
    for fs_val, (cifti_int, _) in SUBCORTICAL_LABELS.items():
        remapped[seg_data == fs_val] = cifti_int

    remap_path = out_dir / 'subcortical_remapped.nii.gz'
    nib.save(nib.Nifti1Image(remapped, seg_img.affine), remap_path)

    # Build wb_command label list file (unique cifti_int → name)
    seen = {}
    for fs_val, (cifti_int, cifti_name) in SUBCORTICAL_LABELS.items():
        seen[cifti_int] = cifti_name

    label_txt = out_dir / 'subcortical_labels.txt'
    with open(label_txt, 'w') as f:
        for cifti_int, cifti_name in sorted(seen.items()):
            f.write(f'{cifti_name}\n{cifti_int} 255 255 255 255\n')

    label_vol = out_dir / 'subcortical_label_vol.nii.gz'
    wb(['-volume-label-import', remap_path, label_txt, label_vol,
        '-drop-unused-labels'])
    return label_vol, remap_path


def extract_subcortical_ts(bold_path: Path, label_mask_path: Path,
                            out_dir: Path, stem: str) -> Path:
    """
    Extract timeseries for subcortical voxels from denoised T1w BOLD,
    saving a masked 4D volume containing only subcortical voxels
    (non-subcortical set to 0) for use in cifti-create-dense-timeseries.
    """
    bold_img  = nib.load(bold_path)
    bold_data = bold_img.get_fdata(dtype=np.float32)   # (x, y, z, t)
    mask_img  = nib.load(label_mask_path)
    mask_data = np.asarray(mask_img.dataobj, dtype=np.int32)

    # Only include voxels whose aparc+aseg label is in SUBCORTICAL_LABELS
    subcort_mask = np.isin(mask_data, list(SUBCORTICAL_LABELS.keys()))
    out_data = np.zeros_like(bold_data)
    out_data[subcort_mask] = bold_data[subcort_mask]

    # Replace NaNs with 0 (wb_command doesn't handle NaN in dense timeseries)
    out_data = np.nan_to_num(out_data, nan=0.0)

    out_path = out_dir / f'{stem}_subcortical_bold.nii.gz'
    nib.save(nib.Nifti1Image(out_data, bold_img.affine, bold_img.header),
             out_path)
    return out_path


# ---------------------------------------------------------------------------
# Per-run surface pipeline
# ---------------------------------------------------------------------------

def process_run(subject: str, run: int, surf_dir: Path,
                label_vol: Path, out_dir: Path) -> Path:
    """
    Run ribbon mapping + fsLR resampling + CIFTI assembly + smoothing.
    Returns path to smoothed per-run CIFTI.
    """
    func_dir    = FMRIPREP / subject / SESSION / 'func'
    fs_surf_dir = FREESURFER / subject / 'surf'
    stem = f'{subject}_{SESSION}_task-{TASK}_run-{run}'

    denoised_path = OUTPUT_ROOT / subject / 'denoised' / \
                    f'{stem}_desc-denoised_bold.nii.gz'
    if not denoised_path.exists():
        raise FileNotFoundError(f'Run 01_denoise.py first: {denoised_path}')

    tmp = out_dir / 'tmp' / f'run-{run}'
    tmp.mkdir(parents=True, exist_ok=True)

    cifti_paths = {}

    for hemi, hemi_fs in [('L', 'lh'), ('R', 'rh')]:
        print(f'  [{hemi}] ribbon mapping ...')
        smoothwm  = surf_dir / f'{subject}_{SESSION}_hemi-{hemi}_smoothwm.surf.gii'
        pial      = surf_dir / f'{subject}_{SESSION}_hemi-{hemi}_pial.surf.gii'
        midthick  = surf_dir / f'{subject}_{SESSION}_hemi-{hemi}_midthickness.surf.gii'
        sphere_reg = fs_surf_dir / f'{hemi_fs}.sphere.reg.surf.gii'

        fsnative_gii = tmp / f'{stem}_hemi-{hemi}_space-fsnative_bold.func.gii'
        fslr_gii     = tmp / f'{stem}_hemi-{hemi}_space-fsLR32k_bold.func.gii'

        # Ribbon-constrained sampling: T1w volume → fsnative surface
        wb(['-volume-to-surface-mapping', denoised_path, midthick,
            fsnative_gii, '-ribbon-constrained', smoothwm, pial])

        # Resample fsnative → fsLR 32k
        wb(['-metric-resample', fsnative_gii, sphere_reg,
            FSLR_SPHERE[hemi], 'ADAP_BARY_AREA', fslr_gii,
            '-area-surfs', midthick, FSLR_MIDTHICK[hemi]])

        cifti_paths[hemi] = fslr_gii

    # Extract subcortical timeseries
    aparcaseg = func_dir / f'{stem}_space-T1w_desc-aparcaseg_dseg.nii.gz'
    subcort_bold = extract_subcortical_ts(denoised_path, aparcaseg, tmp, stem)

    # Assemble CIFTI dtseries
    raw_cifti = tmp / f'{stem}_space-fsLR32k_bold.dtseries.nii'
    n_retained = nib.load(denoised_path).shape[3]
    wb(['-cifti-create-dense-timeseries', raw_cifti,
        '-left-metric',  cifti_paths['L'],
        '-roi-left',     FSLR_ROI['L'],
        '-right-metric', cifti_paths['R'],
        '-roi-right',    FSLR_ROI['R'],
        '-volume',       subcort_bold, label_vol,
        '-timestep',     f'{TR:.4f}'])

    # Smooth CIFTI
    smooth_cifti = out_dir / f'{stem}_space-fsLR32k_bold_smooth.dtseries.nii'
    wb(['-cifti-smoothing', raw_cifti,
        str(SMOOTH_SIGMA), str(SMOOTH_SIGMA), 'COLUMN', smooth_cifti,
        '-left-surface',  FSLR_MIDTHICK['L'],
        '-right-surface', FSLR_MIDTHICK['R']])

    return smooth_cifti


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subject: str) -> None:
    surf_dir = FMRIPREP / subject / SESSION / 'anat'
    out_dir  = OUTPUT_ROOT / subject / 'cifti'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the subcortical label volume once (same anatomy for all runs)
    print(f'[{subject}] Building subcortical label volume ...')
    aparcaseg_run1 = (FMRIPREP / subject / SESSION / 'func' /
                      f'{subject}_{SESSION}_task-{TASK}_run-1'
                      '_space-T1w_desc-aparcaseg_dseg.nii.gz')
    label_vol, _ = make_label_volume(aparcaseg_run1, out_dir)

    run_ciftis = []
    for run in RUNS:
        print(f'\n[{subject}] run-{run}')
        cifti = process_run(subject, run, surf_dir, label_vol, out_dir)
        run_ciftis.append(cifti)

    # Concatenate all runs
    print(f'\n[{subject}] Concatenating {len(run_ciftis)} runs ...')
    concat_cifti = out_dir / f'{subject}_{SESSION}_space-fsLR32k_bold_concat.dtseries.nii'
    merge_args = ['-cifti-merge', concat_cifti]
    for rc in run_ciftis:
        merge_args += ['-cifti', rc]
    wb(merge_args)

    print(f'\nDone. Session CIFTI: {concat_cifti}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python 02_surface_cifti.py sub-XX')
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
Step 1 — Denoise T1w BOLD data (Gordon 2017 protocol).

Denoising model:
  - Friston 24-parameter motion model (6 params at t, t-1, t², (t-1)²)
  - Global signal, white matter, CSF (no derivatives)
  - Scrubbing: FD > 0.2 mm → interpolate → bandpass (0.009-0.08 Hz) → excise
  - CoV masking: high-artefact voxels set to NaN before saving

Usage:
  python 01_denoise.py sub-01
"""
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).parent))
from config import (FMRIPREP, OUTPUT_ROOT, SESSION, TASK, RUNS, TR,
                    FD_THRESHOLD, BP_LOW, BP_HIGH, COV_SD_THRESH)


# ---------------------------------------------------------------------------
# Confound helpers
# ---------------------------------------------------------------------------

def build_friston24(conf: pd.DataFrame) -> np.ndarray:
    """Friston 1996 Volterra expansion: 6(t) + 6(t-1) + 6²(t) + 6²(t-1)."""
    cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    t      = conf[cols].values                            # (n, 6)
    t_lag  = np.vstack([np.zeros((1, 6)), t[:-1]])        # (n, 6) shifted by 1
    t_sq   = t ** 2                                       # (n, 6)
    t_lag_sq = t_lag ** 2                                 # (n, 6)
    return np.hstack([t, t_lag, t_sq, t_lag_sq])          # (n, 24)


def build_confound_matrix(conf: pd.DataFrame) -> np.ndarray:
    """Full confound matrix: Friston 24 + global signal + WM + CSF."""
    motion  = build_friston24(conf)
    physio  = conf[['global_signal', 'white_matter', 'csf']].values
    X = np.hstack([motion, physio])   # (n, 27)
    X = np.nan_to_num(X)              # first-row NaNs from lag → 0
    return X


def get_scrub_mask(conf: pd.DataFrame) -> np.ndarray:
    """Boolean mask: True = retain frame, False = censor (FD > threshold)."""
    fd = conf['framewise_displacement'].fillna(0).values
    return fd <= FD_THRESHOLD


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def regress_confounds(data2d: np.ndarray, X: np.ndarray,
                      retain: np.ndarray) -> np.ndarray:
    """
    OLS regression using only retained frames, applied to all frames.
    data2d: (n_vols, n_voxels), X: (n_vols, n_regressors)
    Returns residuals (n_vols, n_voxels).
    """
    X_int = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.linalg.lstsq(X_int[retain], data2d[retain], rcond=None)[0]
    return data2d - X_int @ beta


def interpolate_censored(data2d: np.ndarray, retain: np.ndarray) -> np.ndarray:
    """
    Linear interpolation across censored frames (Power et al. 2014 approx).
    Produces a continuous signal suitable for bandpass filtering.
    Censored frames are replaced with linearly interpolated values.
    """
    if retain.all():
        return data2d
    n_vols = len(retain)
    t_all   = np.arange(n_vols)
    t_clean = t_all[retain]
    out = data2d.copy()
    for ci in t_all[~retain]:
        before = t_clean[t_clean < ci]
        after  = t_clean[t_clean > ci]
        if len(before) == 0:
            out[ci] = data2d[after[0]]
        elif len(after) == 0:
            out[ci] = data2d[before[-1]]
        else:
            t0, t1 = before[-1], after[0]
            w = (ci - t0) / (t1 - t0)
            out[ci] = (1 - w) * data2d[t0] + w * data2d[t1]
    return out


def bandpass_filter(data2d: np.ndarray, tr: float,
                    lo: float, hi: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter along time axis (axis=0)."""
    nyq = 0.5 / tr
    b, a = butter(2, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, data2d, axis=0)


# ---------------------------------------------------------------------------
# CoV masking
# ---------------------------------------------------------------------------

def compute_cov_mask(data2d: np.ndarray, vol_shape: tuple,
                     smooth_fwhm_mm: float = 5.0,
                     vox_size_mm: float = 2.5) -> np.ndarray:
    """
    Exclude voxels with CoV > local_mean_CoV + COV_SD_THRESH * local_std_CoV.
    Locality defined by Gaussian smoothing (sigma ≈ FWHM/2.355).
    Returns 3D boolean mask (True = keep).
    """
    mean_ts = np.abs(data2d.mean(axis=0))
    mean_ts[mean_ts < 1e-6] = 1e-6      # avoid division by zero
    std_ts  = data2d.std(axis=0)
    cov     = std_ts / mean_ts           # (n_voxels,)

    cov_3d  = cov.reshape(vol_shape)
    sigma   = (smooth_fwhm_mm / 2.355) / vox_size_mm
    cov_smooth = gaussian_filter(cov_3d, sigma=sigma)
    cov_smooth_sq = gaussian_filter(cov_3d ** 2, sigma=sigma)
    local_std = np.sqrt(np.maximum(cov_smooth_sq - cov_smooth ** 2, 0))

    bad_3d = cov_3d > (cov_smooth + COV_SD_THRESH * local_std)
    return ~bad_3d   # True = good voxel


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

def denoise_run(subject: str, run: int, out_dir: Path) -> None:
    func_dir = FMRIPREP / subject / SESSION / 'func'
    stem = f'{subject}_{SESSION}_task-{TASK}_run-{run}'

    bold_path = func_dir / f'{stem}_space-T1w_desc-preproc_bold.nii.gz'
    conf_path = func_dir / f'{stem}_desc-confounds_timeseries.tsv'

    print(f'  Loading BOLD: {bold_path.name}')
    bold_img  = nib.load(bold_path)
    bold_data = bold_img.get_fdata(dtype=np.float32)   # (x, y, z, t)
    vol_shape = bold_data.shape[:3]
    n_vols    = bold_data.shape[3]
    data2d    = bold_data.reshape(-1, n_vols).T        # (t, voxels)

    conf = pd.read_csv(conf_path, sep='\t')

    # --- 1. Build confounds and scrub mask
    X      = build_confound_matrix(conf)
    retain = get_scrub_mask(conf)
    n_censored = (~retain).sum()
    print(f'  Censored frames: {n_censored}/{n_vols} '
          f'({100*n_censored/n_vols:.1f}%)')

    # --- 2. Regress confounds (fit on clean frames, apply to all)
    residuals = regress_confounds(data2d, X, retain)

    # --- 3. Interpolate censored frames → continuous signal for filtering
    residuals_interp = interpolate_censored(residuals, retain)

    # --- 4. Bandpass filter
    filtered = bandpass_filter(residuals_interp, TR, BP_LOW, BP_HIGH)

    # --- 5. Excise censored frames
    clean = filtered[retain]   # (n_retained, n_voxels)

    # --- 6. CoV masking: zero high-artefact voxels, save mask separately
    cov_mask = compute_cov_mask(clean, vol_shape)
    n_bad = (~cov_mask).sum()
    print(f'  CoV-masked voxels: {n_bad} '
          f'({100*n_bad/cov_mask.size:.1f}%)')
    clean[:, ~cov_mask.ravel()] = 0.0   # 0 not NaN — NaN would propagate through wb_command

    # --- 7. Save denoised volume (only retained frames)
    clean_4d = clean.T.reshape(*vol_shape, -1)
    out_img = nib.Nifti1Image(clean_4d, bold_img.affine, bold_img.header)
    out_img.header.set_data_dtype(np.float32)

    out_bold = out_dir / f'{stem}_desc-denoised_bold.nii.gz'
    nib.save(out_img, out_bold)
    print(f'  Saved: {out_bold.name}  ({clean_4d.shape[3]} volumes)')

    # Save spatial CoV mask so downstream steps can reference it
    np.save(out_dir / f'{stem}_desc-covmask.npy', cov_mask)

    # --- 8. Save scrub mask (boolean array, original volume indices)
    np.save(out_dir / f'{stem}_desc-scrubmask.npy', retain)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subject: str) -> None:
    out_dir = OUTPUT_ROOT / subject / 'denoised'
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        print(f'\n[{subject}] run-{run}')
        denoise_run(subject, run, out_dir)
    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python 01_denoise.py sub-XX')
        sys.exit(1)
    main(sys.argv[1])

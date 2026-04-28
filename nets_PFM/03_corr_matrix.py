#!/usr/bin/env python3
"""
Step 3 — Generate parcel-to-parcel correlation matrix from CIFTI timeseries.

Loads the concatenated session CIFTI, applies a parcellation, computes
mean timeseries per parcel, Pearson correlation → Fisher z-transform.

Usage:
  python 03_corr_matrix.py sub-01 --parc /path/to/parcellation.dlabel.nii

Parcellation note:
  Needs a fsLR 32k CIFTI label file (.dlabel.nii) where each grayordinate
  is assigned an integer parcel index (0 = unassigned).

  Suggested sources:
    - Gordon 333:  https://sites.wustl.edu/petersenschlaggarlab/resources/
    - Schaefer 400: https://github.com/ThomasYeoLab/CBIG (also on neuromaps)
    - HCP MMP 360:  available via wb_command -cifti-label-import

  The parcellation must be in the same fsLR 32k space as the CIFTI timeseries.
"""
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_ROOT, SESSION


# ---------------------------------------------------------------------------
# CIFTI helpers
# ---------------------------------------------------------------------------

def load_cifti_timeseries(cifti_path: Path) -> np.ndarray:
    """Load CIFTI dtseries and return data as (n_timepoints, n_grayordinates)."""
    img  = nib.load(cifti_path)
    data = img.get_fdata(dtype=np.float32)
    # nibabel returns CIFTI as (n_timepoints, n_grayordinates)
    return data


def load_parcellation(parc_path: Path) -> np.ndarray:
    """Load CIFTI dlabel and return parcel indices as (n_grayordinates,)."""
    img  = nib.load(parc_path)
    data = img.get_fdata(dtype=np.float32)
    return data.squeeze().astype(int)


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def parcellate(timeseries: np.ndarray, parcel_labels: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Average timeseries within each parcel.
    Returns:
      parcel_ts : (n_timepoints, n_parcels)
      parcel_ids: (n_parcels,) — integer IDs of included parcels
    """
    unique = np.unique(parcel_labels)
    unique = unique[unique > 0]   # 0 = unassigned, skip

    parcel_ts = np.zeros((timeseries.shape[0], len(unique)), dtype=np.float32)
    for i, pid in enumerate(unique):
        mask = parcel_labels == pid
        parcel_ts[:, i] = timeseries[:, mask].mean(axis=1)

    return parcel_ts, unique


def correlation_matrix(parcel_ts: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between all parcel pairs → Fisher z-transform.
    Returns (n_parcels, n_parcels) upper-triangle symmetric matrix.
    """
    # Demean
    ts = parcel_ts - parcel_ts.mean(axis=0, keepdims=True)

    # Pearson r (vectorised)
    norms = np.linalg.norm(ts, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    ts_norm = ts / norms
    r = ts_norm.T @ ts_norm

    # Clip to valid range before Fisher z
    r = np.clip(r, -0.9999, 0.9999)
    np.fill_diagonal(r, 0.0)   # set diagonal to 0, not atanh(1)=inf

    return np.arctanh(r).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subject: str, parc_path: Path) -> None:
    out_dir = OUTPUT_ROOT / subject / 'matrices'
    out_dir.mkdir(parents=True, exist_ok=True)

    cifti_path = (OUTPUT_ROOT / subject / 'cifti' /
                  f'{subject}_{SESSION}_space-fsLR32k_bold_concat.dtseries.nii')
    if not cifti_path.exists():
        raise FileNotFoundError(f'Run 02_surface_cifti.py first: {cifti_path}')

    print(f'[{subject}] Loading CIFTI timeseries ...')
    timeseries = load_cifti_timeseries(cifti_path)
    print(f'  Shape: {timeseries.shape}  '
          f'(timepoints × grayordinates)')

    print(f'[{subject}] Loading parcellation: {parc_path.name}')
    parcel_labels = load_parcellation(parc_path)
    if parcel_labels.shape[0] != timeseries.shape[1]:
        raise ValueError(
            f'Parcellation has {parcel_labels.shape[0]} grayordinates '
            f'but CIFTI has {timeseries.shape[1]}. '
            f'Check that the parcellation is in fsLR 32k space.')

    print(f'[{subject}] Parcellating ...')
    parcel_ts, parcel_ids = parcellate(timeseries, parcel_labels)
    print(f'  {len(parcel_ids)} parcels × {parcel_ts.shape[0]} timepoints')

    print(f'[{subject}] Computing correlation matrix ...')
    cm = correlation_matrix(parcel_ts)

    parc_name = parc_path.stem.split('.')[0]
    out_cm    = out_dir / f'{subject}_{SESSION}_parc-{parc_name}_cm.npy'
    np.save(out_cm, cm)

    out_ids   = out_dir / f'{subject}_{SESSION}_parc-{parc_name}_parcel_ids.npy'
    np.save(out_ids, parcel_ids)

    print(f'  Saved: {out_cm.name}  shape={cm.shape}')
    print(f'  Saved: {out_ids.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='e.g. sub-01')
    parser.add_argument('--parc', required=True,
                        help='Path to fsLR 32k parcellation .dlabel.nii')
    args = parser.parse_args()
    main(args.subject, Path(args.parc))

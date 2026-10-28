#!/usr/bin/env python3
"""
Step 8 — Assign subcortical voxels to cortical networks (winner-take-all).

Subcortical BOLD is too weak for Infomap: with subcortical voxels in the graph
(PFM_VARIANT=subcort, step 03/04), most striatal voxels stay unassigned because
the global density thresholds are filled by strong cortico-cortical edges
(ASD sub-01: 62% of striatal voxels assigned at 5% vs 100% of cortex, ~40%
unassigned in the consensus). Instead, as in Greene et al. 2020 (Neuron), the
cortex is mapped on its own (default cortex-only pipeline, 04 + 04b), and each
subcortical voxel is assigned to the cortical network it correlates with most:

  1. For each voxel: Pearson r with every cortical vertex, Fisher z.
  2. Vertices < CM_DIST_CUTOFF_MM (30 mm) from the voxel are ignored
     (signal bleed from nearby cortex, e.g. putamen ↔ insula).
  3. Per network: mean z over that network's remaining vertices.
  4. Label = network with the highest mean z.

Distances use the subject's own midthickness resampled to the fsLR 32k mesh, so
vertices and voxels are both in T1w space. (The template fsLR coordinates used
in 03/07 are NOT in the voxels' space — fine for cortex-cortex pairs only.)

Inputs:
  cortex labels  BASE_OUTPUT_ROOT/sub-XX/networks/*_consensus_ref-{ref}_communities.npz
                 (default cortex-only run: 04 + 04b)
  timeseries     <BASE_OUTPUT_ROOT>_subcort/sub-XX/cifti/*_bold_concat.dtseries.nii
                 (step 02 with PFM_VARIANT=subcort; its cortex part is identical
                 to the default CIFTI)
Group averages (sub-average*) are not supported: subcortical voxels differ per
subject (native space), so there is no per-voxel average.

Saved (<BASE_OUTPUT_ROOT>_subcort/sub-XX/):
  anat/     *_hemi-{L,R}_space-fsLR32k_desc-native_midthickness.surf.gii
  networks/ *_subcort-wta_ref-{ref}.npz        labels, net_z (voxels × networks),
                                               net_ids, structure, ijk, affine
            *_subcort-wta_ref-{ref}_dseg.nii.gz   label volume (T1w, BOLD grid)
            *_subcort-wta_ref-{ref}_summary.csv   voxels per structure × network

Usage (no PFM_VARIANT needed):
  PFM_DATASET=asd python 08_subcort_assign.py 1
  PFM_DATASET=asd python 08_subcort_assign.py 1 2 3 --ref ca19
"""
import argparse
import subprocess
import sys
import time
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (BASE_OUTPUT_ROOT, FMRIPREP, FREESURFER, SESSION, WB_COMMAND,
                    FSLR_SPHERE, FSLR_ROI, FREESURFER_OVERRIDE, ANAT_DIR_OVERRIDE,
                    CM_DIST_CUTOFF_MM, ATLAS_NETWORK_NAMES, REFERENCES)

SUBCORT_ROOT = BASE_OUTPUT_ROOT.with_name(f'{BASE_OUTPUT_ROOT.name}_subcort')
CHUNK  = 1000        # voxels per correlation chunk
R_CLIP = 0.999999    # keep arctanh finite


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f'{s/60:.1f} min' if s >= 60 else f'{s:.1f} s'


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def native_cortex_coords(subject: str) -> np.ndarray:
    """
    (n_cortex, 3) T1w-space coordinates of the CIFTI cortex vertices: the
    subject's fMRIPrep midthickness resampled to fsLR 32k (same sphere.reg
    route as the timeseries in step 02), medial wall removed, L then R.
    """
    anat_in  = ANAT_DIR_OVERRIDE.get(subject, FMRIPREP / subject / SESSION / 'anat')
    fs_surf  = FREESURFER_OVERRIDE.get(subject, FREESURFER) / subject / 'surf'
    anat_out = SUBCORT_ROOT / subject / 'anat'
    anat_out.mkdir(parents=True, exist_ok=True)

    coords = []
    for hemi, hemi_fs in [('L', 'lh'), ('R', 'rh')]:
        out = anat_out / f'{subject}_{SESSION}_hemi-{hemi}_space-fsLR32k_desc-native_midthickness.surf.gii'
        if not out.exists():
            subprocess.run([WB_COMMAND, '-surface-resample',
                            str(anat_in / f'{subject}_{SESSION}_hemi-{hemi}_midthickness.surf.gii'),
                            str(fs_surf / f'{hemi_fs}.sphere.reg.surf.gii'),
                            str(FSLR_SPHERE[hemi]), 'BARYCENTRIC', str(out)], check=True)
        roi = nib.load(FSLR_ROI[hemi]).darrays[0].data.astype(bool)
        coords.append(nib.load(out).darrays[0].data[roi])
    return np.concatenate(coords).astype(np.float32)


def load_timeseries(subject: str):
    """
    Split the subcort concat CIFTI into z-normalised (unit-norm) cortex and
    subcortical blocks. Returns cortex (T, n_cortex), subcort (T, n_vox), the
    per-voxel structure names, ijk indices, and the volume affine/shape.
    """
    path = (SUBCORT_ROOT / subject / 'cifti' /
            f'{subject}_{SESSION}_space-fsLR32k_bold_concat.dtseries.nii')
    if not path.exists():
        raise FileNotFoundError(f'Run 02 with PFM_VARIANT=subcort first: {path}')
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    bm   = img.header.get_axis(1)

    data -= data.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms[norms < 1e-10] = 1.0
    data /= norms

    n_cortex  = int(sum(s.stop - s.start for n, s, _ in bm.iter_structures()
                        if n.startswith('CIFTI_STRUCTURE_CORTEX')))
    structure = np.array([n.removeprefix('CIFTI_STRUCTURE_')
                          for n in bm.name[n_cortex:]])
    ijk = bm.voxel[n_cortex:]
    return (data[:, :n_cortex], data[:, n_cortex:], structure, ijk,
            bm.affine, bm.volume_shape)


def load_cortex_labels(subject: str, ref: str, n_cortex: int) -> np.ndarray:
    """(n_cortex,) network label per cortex vertex from the default run (0 = none)."""
    stem = f'{subject}_{SESSION}_space-fsLR32k'
    valid = np.load(BASE_OUTPUT_ROOT / subject / 'cm' / f'{stem}_cm_meta.npz')['valid_mask']
    net   = np.load(BASE_OUTPUT_ROOT / subject / 'networks' /
                    f'{stem}_consensus_ref-{ref}_communities.npz')['network_labels']
    if len(valid) != n_cortex:
        raise ValueError(f'{subject}: default run has {len(valid)} grayordinates, '
                         f'subcort CIFTI cortex has {n_cortex}')
    labels = np.zeros(n_cortex, dtype=np.int32)
    labels[valid] = net
    return labels


# ---------------------------------------------------------------------------
# Winner-take-all
# ---------------------------------------------------------------------------

def network_z(cortex: np.ndarray, subcort: np.ndarray, labels: np.ndarray,
              ctx_xyz: np.ndarray, vox_xyz: np.ndarray):
    """
    Mean Fisher z between each voxel and each network's vertices, ignoring
    vertices < CM_DIST_CUTOFF_MM away. Returns (net_ids, net_z (n_vox, n_nets)).
    """
    net_ids = np.unique(labels[labels > 0])
    labelled = np.where(labels > 0)[0]
    # one-hot vertex → network (only labelled vertices)
    onehot = sp.csr_matrix((np.ones(len(labelled), dtype=np.float32),
                            (np.arange(len(labelled)), np.searchsorted(net_ids, labels[labelled]))),
                           shape=(len(labelled), len(net_ids)))
    ctx = cortex[:, labelled]
    ctx_xyz = ctx_xyz[labelled]

    net_z = np.full((subcort.shape[1], len(net_ids)), np.nan, dtype=np.float32)
    for start in range(0, subcort.shape[1], CHUNK):
        end = min(start + CHUNK, subcort.shape[1])
        z = np.arctanh(np.clip(subcort[:, start:end].T @ ctx, -R_CLIP, R_CLIP))
        far = cdist(vox_xyz[start:end], ctx_xyz) >= CM_DIST_CUTOFF_MM
        z *= far
        n = far.astype(np.float32) @ onehot          # vertices counted per network
        with np.errstate(invalid='ignore', divide='ignore'):
            net_z[start:end] = (z @ onehot) / n      # n = 0 → nan
    return net_ids, net_z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subject: str, refs: list[str]) -> None:
    t0 = time.perf_counter()
    print(f'[{subject}] Loading subcort CIFTI + native cortex coordinates ...')
    cortex, subcort, structure, ijk, affine, vol_shape = load_timeseries(subject)
    ctx_xyz = native_cortex_coords(subject)
    vox_xyz = nib.affines.apply_affine(affine, ijk).astype(np.float32)
    flat    = subcort.std(axis=0) == 0               # e.g. CoV-masked voxels
    print(f'  {cortex.shape[0]} frames, {cortex.shape[1]} vertices, '
          f'{subcort.shape[1]} voxels ({flat.sum()} zero-variance)')

    # Sanity check that both are in the same space: putamen ↔ nearest cortex
    put = np.char.startswith(structure, 'PUTAMEN')
    if put.any():
        d = cdist(vox_xyz[put], ctx_xyz).min(axis=1)
        print(f'  putamen → nearest cortex vertex: median {np.median(d):.1f} mm '
              f'(expect ~5–15 mm; insula)')

    out_dir = SUBCORT_ROOT / subject / 'networks'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{subject}_{SESSION}_space-fsLR32k'

    for ref in refs:
        t = time.perf_counter()
        labels = load_cortex_labels(subject, ref, cortex.shape[1])
        net_ids, net_z = network_z(cortex, subcort, labels, ctx_xyz, vox_xyz)
        vox_labels = np.where(np.isnan(net_z).all(axis=1), 0,
                              net_ids[np.nanargmax(np.nan_to_num(net_z, nan=-np.inf), axis=1)])
        vox_labels[flat] = 0

        out = out_dir / f'{stem}_subcort-wta_ref-{ref}'
        np.savez(f'{out}.npz', labels=vox_labels, net_z=net_z, net_ids=net_ids,
                 structure=structure, ijk=ijk, affine=affine)

        vol = np.zeros(vol_shape, dtype=np.int16)
        vol[tuple(ijk.T)] = vox_labels
        nib.save(nib.Nifti1Image(vol, affine), f'{out}_dseg.nii.gz')

        names = ATLAS_NETWORK_NAMES[ref]
        summary = (pd.crosstab(structure, [names[int(l)] for l in vox_labels])
                     .rename_axis(index='structure', columns='network'))
        summary.to_csv(f'{out}_summary.csv')
        print(f'\n[{subject}] ref={ref}  [{_elapsed(t)}]')
        print(summary.to_string())

    print(f'\n[{subject}] Total time: {_elapsed(t0)}')


def parse_subject(s: str) -> str:
    """1 / 01 / sub-01 → sub-01; non-numeric labels kept (e.g. average → sub-average)."""
    label = s.removeprefix('sub-')
    return f'sub-{int(label):02d}' if label.isdigit() else f'sub-{label}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subjects', nargs='+', help='subject ids: 1, 01, or sub-01')
    parser.add_argument('--ref', nargs='+', default=REFERENCES,
                        help='Reference(s) whose 04b cortex labels to use (default: config.REFERENCES)')
    args = parser.parse_args()
    for s in args.subjects:
        main(parse_subject(s), args.ref)

#!/usr/bin/env python3
"""
Step 7 — Group-average vertex-wise correlation matrix.

Builds a group-average connectome from the individual concatenated CIFTI
timeseries (step 02 output) and thresholds it exactly like 03_vertex_cm.py,
so the result can go through 04_infomap.py / 04b_relabel.py / 05_plot.py as a
pseudo-subject (default label: sub-average). Intended use: a dataset-specific
group network atlas, e.g. as a reference for 04b (cf. ca19magjudgeDDnr for DNumRisk).

Why not average the step-03 outputs: those are per-subject *thresholded* sparse
matrices (each with its own r-cutoff, local pairs zeroed) — a 0 there means
"below this subject's cutoff", not r = 0, so their mean is not a mean correlation.

Averaging: per row-chunk, each subject's Pearson r is Fisher-z transformed,
averaged across subjects (equal weight per subject, regardless of frame count),
and back-transformed. Then, as in step 03: same-hemisphere surface pairs and
pairs involving a subcortical voxel < 30 mm zeroed, r-cutoffs per density estimated from 100k random pairs (same
seed/sampling as step 03), one sparse CSR per density. The dense matrix is
never stored.

Subjects: all with a concat CIFTI, or one group (--group, from config.get_groups).
Node set: grayordinates valid (non-zero variance) in ALL included subjects.

Memory: all subjects' normalised timeseries are held in RAM
(~59k nodes × ~1,400 frames × 4 B ≈ 0.33 GB per ASD subject → ~33 GB for 98).
Run on a large-RAM VM.

Saved files (OUTPUT_ROOT/sub-<label>/cm/), same naming as step 03:
  sub-<label>_ses-1_space-fsLR32k_density-{d}_cm.npz
  sub-<label>_ses-1_space-fsLR32k_cm_meta.npz
  sub-<label>_subjects.txt                          (subjects included)

Usage:
  python 07_group_avg_cm.py                         # all subjects with a concat CIFTI
  python 07_group_avg_cm.py --exclude 5 17
  python 07_group_avg_cm.py --group ASD             # → sub-average-ASD
  python 07_group_avg_cm.py --group Control         # → sub-average-Control
  (groups/names from config.get_groups / GROUP_NAMES: ASD/Control, DD/TD)
  python 07_group_avg_cm.py --subjects 1 2 3 --label average-test
then:
  python 04_infomap.py average
  python 04b_relabel.py average --ref <atlas.npz> --ref-name <name>
  python 05_plot.py average
"""
import argparse
import importlib.util
import sys
import time
import numpy as np
import nibabel as nib
import numexpr as ne
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (OUTPUT_ROOT, SESSION, ALL_SUBJECTS, GROUP_NAMES, get_groups,
                    EXCLUDE_SUBJECTS, INCLUDE_SUBCORTICAL,
                    INFOMAP_DENSITIES, CM_CHUNK_SIZE, CM_DIST_CUTOFF_MM)

# get_node_info from step 03 (module name starts with a digit → importlib)
_spec = importlib.util.spec_from_file_location(
    'vertex_cm', Path(__file__).parent / '03_vertex_cm.py')
_vertex_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vertex_cm)
get_node_info = _vertex_cm.get_node_info

R_CLIP = 0.999999   # keep arctanh finite


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f'{s/60:.1f} min' if s >= 60 else f'{s:.1f} s'


def concat_cifti(subject: str) -> Path:
    return (OUTPUT_ROOT / subject / 'cifti' /
            f'{subject}_{SESSION}_space-fsLR32k_bold_concat.dtseries.nii')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_normalised(subjects: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Load each subject's concatenated CIFTI, z-normalise per node (zero mean,
    unit norm → dot product = Pearson r). Returns the list of (n_t, n_nodes)
    arrays restricted to nodes valid in every subject, plus that valid mask.
    """
    ts_list, valid = [], None
    for k, subject in enumerate(subjects, 1):
        t = time.perf_counter()
        data = nib.load(concat_cifti(subject)).get_fdata(dtype=np.float32)
        data -= data.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(data, axis=0, keepdims=True)
        ok = norms[0] > 1e-6
        norms[norms < 1e-10] = 1.0
        data /= norms
        valid = ok if valid is None else (valid & ok)
        ts_list.append(data)
        print(f'  [{k}/{len(subjects)}] {subject}: {data.shape[0]} frames, '
              f'{(~ok).sum()} zero-variance nodes  [{_elapsed(t)}]', flush=True)

    if not valid.all():
        ts_list = [ts[:, valid] for ts in ts_list]   # restrict to common nodes
    return ts_list, valid


# ---------------------------------------------------------------------------
# Fisher-z averaged correlations
# ---------------------------------------------------------------------------

def mean_r_pairs(ts_list: list[np.ndarray], i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Group-average r for node pairs (i[k], j[k]) via Fisher z."""
    zsum = np.zeros(len(i), dtype=np.float64)
    for ts in ts_list:
        r = np.clip((ts[:, i] * ts[:, j]).sum(axis=0), -R_CLIP, R_CLIP)
        zsum += np.arctanh(r)
    return np.tanh(zsum / len(ts_list)).astype(np.float32)


def mean_r_chunk(ts_list: list[np.ndarray], start: int, end: int) -> np.ndarray:
    """Group-average r for rows start:end against all nodes, via Fisher z."""
    n_nodes = ts_list[0].shape[1]
    zsum = np.zeros((end - start, n_nodes), dtype=np.float32)
    for ts in ts_list:
        r = ts[:, start:end].T @ ts
        ne.evaluate('zsum + arctanh(where(r > R_CLIP, R_CLIP, where(r < -R_CLIP, -R_CLIP, r)))',
                    local_dict={'zsum': zsum, 'r': r, 'R_CLIP': np.float32(R_CLIP)},
                    out=zsum)
    n = np.float32(len(ts_list))
    return ne.evaluate('tanh(zsum / n)')


# ---------------------------------------------------------------------------
# Thresholds + chunked sparse CMs (mirrors 03_vertex_cm.py)
# ---------------------------------------------------------------------------

def estimate_thresholds(ts_list, coords, hemi_ids, target_densities,
                        n_sample: int = 100_000) -> dict:
    """As 03_vertex_cm.estimate_thresholds, but on group-average r."""
    n_nodes = coords.shape[0]
    rng = np.random.default_rng(42)

    i_s = rng.integers(0, n_nodes, n_sample * 2)
    j_s = rng.integers(0, n_nodes, n_sample * 2)
    keep = j_s > i_s
    i_s, j_s = i_s[keep][:n_sample], j_s[keep][:n_sample]

    dist_s   = np.linalg.norm(coords[i_s] - coords[j_s], axis=1)
    same_hem = hemi_ids[i_s] == hemi_ids[j_s]
    both_srf = (hemi_ids[i_s] < 2) & (hemi_ids[j_s] < 2)
    is_local = (dist_s < CM_DIST_CUTOFF_MM) & (same_hem | ~both_srf)

    local_frac = is_local.mean()
    print(f'  Estimated local-pair fraction: {100*local_frac:.1f}%')

    nl = ~is_local
    r_sample = mean_r_pairs(ts_list, i_s[nl], j_s[nl])

    n_total = n_nodes * (n_nodes - 1) // 2
    n_nonlocal = int(n_total * (1 - local_frac))

    thresholds = {}
    for d in target_densities:
        n_target = int(d * n_total)
        if n_target >= n_nonlocal:
            thresholds[d] = float('-inf')
        else:
            pct = (1.0 - n_target / n_nonlocal) * 100.0
            thresholds[d] = float(np.percentile(r_sample, pct))
        print(f'  density={d:.3f}  r_cutoff={thresholds[d]:.4f}  '
              f'(~{n_target/1e6:.1f}M edges)')
    return thresholds


def compute_sparse_cms(ts_list, coords, hemi_ids, thresholds,
                       chunk_size: int = CM_CHUNK_SIZE) -> dict:
    """As 03_vertex_cm.compute_sparse_cms, with r = group-average r per chunk."""
    n_nodes = coords.shape[0]
    acc = {d: ([], [], []) for d in thresholds}
    j_all = np.arange(n_nodes, dtype=np.int32)
    n_chunks = (n_nodes + chunk_size - 1) // chunk_size
    t0 = time.perf_counter()

    for k, start in enumerate(range(0, n_nodes, chunk_size), 1):
        end = min(start + chunk_size, n_nodes)
        r = mean_r_chunk(ts_list, start, end)

        dist = cdist(coords[start:end], coords)
        same_hem = hemi_ids[start:end, None] == hemi_ids[None, :]
        both_srf = (hemi_ids[start:end] < 2)[:, None] & (hemi_ids < 2)[None, :]
        r[(dist < CM_DIST_CUTOFF_MM) & (same_hem | ~both_srf)] = 0.0

        upper = j_all[None, :] > np.arange(start, end, dtype=np.int32)[:, None]
        for d, cutoff in thresholds.items():
            keep = upper & (r > cutoff)
            chunk_r, chunk_c = np.where(keep)
            rows_d, cols_d, vals_d = acc[d]
            rows_d.append((chunk_r + start).astype(np.int32))
            cols_d.append(chunk_c.astype(np.int32))
            vals_d.append(r[keep].astype(np.float32))

        if k == 1 or k % 10 == 0 or k == n_chunks:
            eta = (time.perf_counter() - t0) / k * (n_chunks - k) / 60
            print(f'  chunk {k}/{n_chunks}  (ETA {eta:.0f} min)', flush=True)

    return {d: (np.concatenate(rl), np.concatenate(cl), np.concatenate(vl))
            for d, (rl, cl, vl) in acc.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subjects: list[str], label: str) -> None:
    t_total = time.perf_counter()
    group = f'sub-{label}'
    out_dir = OUTPUT_ROOT / group / 'cm'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[{group}] Loading {len(subjects)} subjects ...')
    ts_list, valid = load_normalised(subjects)
    n_nodes = int(valid.sum())
    gb = sum(ts.nbytes for ts in ts_list) / 1e9
    print(f'  Common valid nodes: {n_nodes}/{len(valid)}  '
          f'({gb:.1f} GB of timeseries in RAM)')

    coords, hemi_ids = get_node_info(concat_cifti(subjects[0]))
    coords, hemi_ids = coords[valid], hemi_ids[valid]

    t = time.perf_counter()
    print(f'[{group}] Estimating correlation thresholds ...')
    thresholds = estimate_thresholds(ts_list, coords, hemi_ids, INFOMAP_DENSITIES)
    print(f'  [{_elapsed(t)}]')

    t = time.perf_counter()
    print(f'[{group}] Computing group-average correlations '
          f'(chunk_size={CM_CHUNK_SIZE}) ...')
    sparse_cms = compute_sparse_cms(ts_list, coords, hemi_ids, thresholds)
    print(f'  Correlation computation done  [{_elapsed(t)}]')

    stem = f'{group}_{SESSION}_space-fsLR32k'
    for d, (rows, cols, vals) in sparse_cms.items():
        csr = sp.csr_matrix((np.concatenate([vals, vals]),
                             (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                            shape=(n_nodes, n_nodes))
        d_str = f'{d:.3f}'.replace('0.', '')
        out_path = out_dir / f'{stem}_density-{d_str}_cm.npz'
        sp.save_npz(out_path, csr)
        print(f'  Saved density={d:.3f}: {csr.nnz//2:,} edges  → {out_path.name}')

    np.savez(out_dir / f'{stem}_cm_meta.npz',
             coords=coords, hemi_ids=hemi_ids, valid_mask=valid,
             n_nodes_original=np.array([len(valid)]))
    (OUTPUT_ROOT / group / f'{group}_subjects.txt').write_text('\n'.join(subjects) + '\n')
    print(f'  Saved node metadata + subject list')
    print(f'\n[{group}] Total time: {_elapsed(t_total)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject numbers (default: all with a concat CIFTI)')
    parser.add_argument('--exclude', nargs='+', type=int, default=[])
    parser.add_argument('--group', default='all',
                        choices=['all', *GROUP_NAMES.values()],
                        help='Restrict to one group from config.get_groups() (default: all; '
                             'subjects without a group label are only included with all)')
    parser.add_argument('--label', default=None,
                        help="Output label → sub-<label> (default 'average', "
                             "or 'average-<group>' with --group)")
    args = parser.parse_args()
    if INCLUDE_SUBCORTICAL:
        # Subcortical voxels are in each subject's native space (different voxel
        # sets) → no per-node correspondence to average over.
        sys.exit('07 is cortex-only: unset PFM_VARIANT (subcortical → 08_subcort_assign.py)')

    candidates = args.subjects or ALL_SUBJECTS
    excluded = set(args.exclude) | set(EXCLUDE_SUBJECTS)
    for n in sorted(set(candidates) & set(EXCLUDE_SUBJECTS)):
        print(f'Excluding sub-{n:02d} (config): {EXCLUDE_SUBJECTS[n]}')
    subjects = [f'sub-{n:02d}' for n in candidates if n not in excluded]
    if args.group != 'all':
        group_id = {v: k for k, v in GROUP_NAMES.items()}[args.group]
        groups = get_groups()
        subjects = [s for s in subjects if groups.get(s) == group_id]
    label = args.label or ('average' if args.group == 'all' else f'average-{args.group}')
    missing = [s for s in subjects if not concat_cifti(s).exists()]
    if missing and args.subjects:
        sys.exit(f'Missing concat CIFTI for: {missing}')
    if missing:
        print(f'Skipping {len(missing)} subjects without concat CIFTI: {missing}')
    subjects = [s for s in subjects if s not in missing]
    if not subjects:
        sys.exit('No subjects selected.')
    print(f'{len(subjects)} subjects (group: {args.group}) → sub-{label}')
    main(subjects, label)

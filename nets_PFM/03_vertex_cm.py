#!/usr/bin/env python3
"""
Step 3 — Vertex-wise correlation matrix (Gordon 2017 protocol).

Computes the all-to-all cross-correlation matrix of the CIFTI timeseries
(cortical surface vertices + subcortical voxels), following Gordon 2017:
  - Correlations between nodes within 30 mm (same-hemisphere surface, or
    any pair involving subcortical) are set to zero.
  - For within-hemisphere surface connections, distance is approximated
    as Euclidean between midthickness coordinates (true geodesic would
    require a much more expensive computation; the approximation is
    conservative: Euclidean ≤ geodesic, so we zero slightly fewer pairs).
  - The matrix is thresholded at each target density and saved as a
    scipy sparse matrix (.npz), one file per density threshold.

The full dense matrix (~90k × 90k = 33 GB) is never materialised.
Computation proceeds in row-chunks of CM_CHUNK_SIZE nodes.

Memory estimate per chunk (CM_CHUNK_SIZE=200, n_nodes~85k):
  r_chunk   200 × 85k × float32  =  68 MB
  dist      200 × 85k × float32  =  68 MB
  masks     200 × 85k × bool     =  17 MB each
  Total                          ~ 175 MB peak per chunk

Saved files (OUTPUT_ROOT/sub-XX/cm/):
  sub-XX_ses-1_space-fsLR32k_density-{d}_cm.npz   (scipy sparse CSR)
  sub-XX_ses-1_space-fsLR32k_cm_meta.npz           (node coords, hemi_ids,
                                                     valid_mask, n_nodes_total)

Usage:
  python 03_vertex_cm.py sub-01
"""
import sys
import time
import numpy as np
import nibabel as nib
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (OUTPUT_ROOT, SESSION, FSLR_MIDTHICK,
                    INFOMAP_DENSITIES, CM_CHUNK_SIZE, CM_DIST_CUTOFF_MM)


# ---------------------------------------------------------------------------
# Node coordinate extraction
# ---------------------------------------------------------------------------

def get_node_info(cifti_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D coordinates (mm) and hemisphere IDs for every grayordinate
    in the CIFTI file, using the embedded brain model metadata.

    hemi_ids encoding: 0 = left cortex, 1 = right cortex, 2 = subcortical

    Returns:
      coords   : (n_nodes, 3) float32
      hemi_ids : (n_nodes,)   int8
    """
    img     = nib.load(cifti_path)
    bm_axis = img.header.get_axis(1)
    n_nodes = bm_axis.size

    coords   = np.zeros((n_nodes, 3), dtype=np.float32)
    hemi_ids = np.full(n_nodes, 2, dtype=np.int8)

    surf_coords = {}
    for key, path in [('CIFTI_STRUCTURE_CORTEX_LEFT',  FSLR_MIDTHICK['L']),
                      ('CIFTI_STRUCTURE_CORTEX_RIGHT', FSLR_MIDTHICK['R'])]:
        surf_coords[key] = nib.load(path).agg_data()[0].astype(np.float32)

    affine = bm_axis.affine   # voxel-ijk → mm for subcortical structures

    for struct_name, slc, model in bm_axis.iter_structures():
        if struct_name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            coords[slc]   = surf_coords[struct_name][model.vertex]
            hemi_ids[slc] = 0
        elif struct_name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            coords[slc]   = surf_coords[struct_name][model.vertex]
            hemi_ids[slc] = 1
        else:
            ijk = model.voxel.astype(np.float32)
            xyz = nib.affines.apply_affine(affine, ijk).astype(np.float32)
            coords[slc] = xyz
            # hemi_ids stays 2 (subcortical)

    return coords, hemi_ids


# ---------------------------------------------------------------------------
# Threshold estimation (random sampling)
# ---------------------------------------------------------------------------

def estimate_thresholds(ts_norm: np.ndarray, coords: np.ndarray,
                        hemi_ids: np.ndarray,
                        target_densities: list,
                        n_sample: int = 100_000) -> dict:
    """
    Estimate the correlation threshold r_cutoff for each target density.

    Strategy:
      1. Sample n_sample random upper-triangle pairs.
      2. Compute their distances and mark local pairs (< 30 mm, same-hemi surface).
      3. Compute Pearson r for non-local pairs.
      4. Estimate what fraction of ALL pairs are local (local_frac).
      5. For each density d:
           n_edges_target   = d × n_total_pairs
           n_nonlocal_pairs ≈ n_total_pairs × (1 - local_frac)
           fraction to keep from non-local = n_edges_target / n_nonlocal_pairs
           r_cutoff = (1 - fraction_to_keep)-th percentile of non-local r values
    """
    n_nodes = ts_norm.shape[1]
    rng = np.random.default_rng(42)

    i_s = rng.integers(0, n_nodes, n_sample * 2)
    j_s = rng.integers(0, n_nodes, n_sample * 2)
    keep = j_s > i_s
    i_s, j_s = i_s[keep][:n_sample], j_s[keep][:n_sample]

    dist_s   = np.linalg.norm(coords[i_s] - coords[j_s], axis=1)
    same_hem = hemi_ids[i_s] == hemi_ids[j_s]
    both_srf = (hemi_ids[i_s] < 2) & (hemi_ids[j_s] < 2)
    is_local = (dist_s < CM_DIST_CUTOFF_MM) & same_hem & both_srf

    local_frac = is_local.mean()
    print(f'  Estimated local-pair fraction: {100*local_frac:.1f}%')

    # Correlations for non-local pairs
    nl = ~is_local
    r_sample = (ts_norm[:, i_s[nl]] * ts_norm[:, j_s[nl]]).sum(axis=0)

    n_total = n_nodes * (n_nodes - 1) // 2
    n_nonlocal = int(n_total * (1 - local_frac))

    thresholds = {}
    for d in target_densities:
        n_target = int(d * n_total)
        if n_target >= n_nonlocal:
            thresholds[d] = float('-inf')
        else:
            frac_keep = n_target / n_nonlocal
            pct = (1.0 - frac_keep) * 100.0
            thresholds[d] = float(np.percentile(r_sample, pct))
        print(f'  density={d:.3f}  r_cutoff={thresholds[d]:.4f}  '
              f'(~{n_target/1e6:.1f}M edges)')

    return thresholds


# ---------------------------------------------------------------------------
# Chunked correlation computation
# ---------------------------------------------------------------------------

def compute_sparse_cms(ts_norm: np.ndarray, coords: np.ndarray,
                       hemi_ids: np.ndarray, thresholds: dict,
                       chunk_size: int = CM_CHUNK_SIZE) -> dict:
    """
    Compute the upper-triangle correlation matrix in row-chunks.
    For each chunk:
      - Compute r (chunk_size × n_nodes) via dot product of normalised series.
      - Zero local connections (within 30 mm, same-hemisphere surface).
      - Retain only upper-triangle entries (col > row).
      - For each density, accumulate entries above the threshold.

    Returns dict: {density: (rows_arr, cols_arr, vals_arr)}
    """
    n_nodes  = ts_norm.shape[1]
    min_cut  = min(thresholds.values())

    # Accumulators per density
    acc = {d: ([], [], []) for d in thresholds}

    j_all = np.arange(n_nodes, dtype=np.int32)

    for start in range(0, n_nodes, chunk_size):
        end     = min(start + chunk_size, n_nodes)
        cs      = end - start
        print(f'\r  chunk {start//chunk_size + 1}/'
              f'{(n_nodes + chunk_size - 1)//chunk_size}  '
              f'nodes {start}-{end}', end='', flush=True)

        # Correlations: (cs × n_nodes)
        r = (ts_norm[:, start:end].T @ ts_norm)   # already unit-norm

        # Zero local connections (same-hemisphere surface within 30 mm)
        dist = cdist(coords[start:end], coords)    # (cs × n_nodes) float64
        same_hem = hemi_ids[start:end, None] == hemi_ids[None, :]
        both_srf = (hemi_ids[start:end] < 2)[:, None] & (hemi_ids < 2)[None, :]
        local    = (dist < CM_DIST_CUTOFF_MM) & same_hem & both_srf
        r[local] = 0.0

        # Upper triangle mask (global col > global row)
        i_global = np.arange(start, end, dtype=np.int32)[:, None]  # (cs, 1)
        upper    = j_all[None, :] > i_global                         # (cs, n_nodes)

        # Accumulate per density threshold
        for d, cutoff in thresholds.items():
            keep = upper & (r > cutoff)
            chunk_r, chunk_c = np.where(keep)
            rows_d, cols_d, vals_d = acc[d]
            rows_d.append((chunk_r + start).astype(np.int32))
            cols_d.append(chunk_c.astype(np.int32))
            vals_d.append(r[keep].astype(np.float32))

    print()  # newline after progress

    # Concatenate and return
    result = {}
    for d, (rl, cl, vl) in acc.items():
        result[d] = (np.concatenate(rl), np.concatenate(cl), np.concatenate(vl))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f'{s/60:.1f} min' if s >= 60 else f'{s:.1f} s'


def main(subject: str) -> None:
    t_total = time.perf_counter()

    cifti_path = (OUTPUT_ROOT / subject / 'cifti' /
                  f'{subject}_{SESSION}_space-fsLR32k_bold_concat.dtseries.nii')
    if not cifti_path.exists():
        raise FileNotFoundError(f'Run 02_surface_cifti.py first: {cifti_path}')

    out_dir = OUTPUT_ROOT / subject / 'cm'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load CIFTI
    t = time.perf_counter()
    print(f'[{subject}] Loading CIFTI timeseries ...')
    img  = nib.load(cifti_path)
    data = img.get_fdata(dtype=np.float32)   # (n_t, n_nodes)
    print(f'  Shape: {data.shape}  (timepoints × grayordinates)  [{_elapsed(t)}]')

    # ---- Extract node coordinates
    t = time.perf_counter()
    print(f'[{subject}] Extracting node coordinates ...')
    coords, hemi_ids = get_node_info(cifti_path)

    # ---- Exclude zero-variance nodes (medial wall, dropout voxels)
    std = data.std(axis=0)
    valid = std > 1e-6
    n_excluded = (~valid).sum()
    print(f'  Zero-variance nodes excluded: {n_excluded} '
          f'({100*n_excluded/len(valid):.1f}%)  [{_elapsed(t)}]')

    data      = data[:, valid]
    coords    = coords[valid]
    hemi_ids  = hemi_ids[valid]
    n_nodes   = data.shape[1]
    print(f'  Valid nodes: {n_nodes}')

    # ---- Normalise: zero-mean, unit-norm per node (→ dot product = Pearson r)
    data -= data.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms[norms < 1e-10] = 1.0
    ts_norm = (data / norms).astype(np.float32)

    # ---- Estimate r thresholds for each target density
    t = time.perf_counter()
    print(f'[{subject}] Estimating correlation thresholds ...')
    thresholds = estimate_thresholds(ts_norm, coords, hemi_ids,
                                     INFOMAP_DENSITIES)
    print(f'  [{_elapsed(t)}]')

    # ---- Compute correlation matrix in chunks
    t = time.perf_counter()
    print(f'[{subject}] Computing vertex-wise correlations '
          f'(chunk_size={CM_CHUNK_SIZE}) ...')
    sparse_cms = compute_sparse_cms(ts_norm, coords, hemi_ids, thresholds)
    print(f'  Correlation computation done  [{_elapsed(t)}]')

    # ---- Save one sparse CSR matrix per density
    stem = f'{subject}_{SESSION}_space-fsLR32k'
    for d, (rows, cols, vals) in sparse_cms.items():
        # Symmetric: add lower triangle
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
        all_vals = np.concatenate([vals, vals])
        csr = sp.csr_matrix((all_vals, (all_rows, all_cols)),
                            shape=(n_nodes, n_nodes))
        d_str = f'{d:.3f}'.replace('0.', '')
        out_path = out_dir / f'{stem}_density-{d_str}_cm.npz'
        sp.save_npz(out_path, csr)
        print(f'  Saved density={d:.3f}: {csr.nnz//2:,} edges  → {out_path.name}')

    # ---- Save node metadata (needed by 04_infomap.py)
    meta_path = out_dir / f'{stem}_cm_meta.npz'
    np.savez(meta_path,
             coords=coords,
             hemi_ids=hemi_ids,
             valid_mask=valid,
             n_nodes_original=np.array([len(valid)]))
    print(f'  Saved node metadata → {meta_path.name}')
    print(f'\n[{subject}] Total time: {_elapsed(t_total)}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python 03_vertex_cm.py sub-XX')
        sys.exit(1)
    main(sys.argv[1])

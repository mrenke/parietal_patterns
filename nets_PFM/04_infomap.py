#!/usr/bin/env python3
"""
Step 4 — Infomap network detection (Gordon 2017 protocol).

Runs the Infomap community detection algorithm on the thresholded
vertex-wise correlation matrix produced by 03_vertex_cm.py.

Gordon 2017 protocol:
  - Two-level Infomap on undirected weighted graph
  - Thresholds from 0.3% to 5% density
  - Small communities (< 400 nodes) removed
  - Consensus assignment by collapsing across density thresholds

Reference-atlas labelling is done separately in 04b_relabel.py.

Saved files (OUTPUT_ROOT/sub-XX/networks/):
  sub-XX_ses-1_space-fsLR32k_density-{d}_communities.npz  → modules, density
  sub-XX_ses-1_space-fsLR32k_consensus_communities.npz     → modules

Usage:
  python 04_infomap.py sub-01
  python 04_infomap.py sub-01 --density 0.005
"""
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_ROOT, SESSION, INFOMAP_DENSITIES, FSLR_ROI, FSLR_MIDTHICK

MIN_COMMUNITY_SIZE = 400   # communities smaller than this are marked unassigned
MIN_PATCH_AREA_MM2 = 30.0  # spatial patches smaller than this are removed + dilated (Gordon 2017)


# ---------------------------------------------------------------------------
# Infomap runner
# ---------------------------------------------------------------------------

def run_infomap(csr: sp.csr_matrix, n_trials: int = 10) -> np.ndarray:
    """
    Run Infomap on the sparse symmetric graph.
    Returns array of module ids, one per node (1-indexed, 0 = unassigned).
    """
    from infomap import Infomap

    im = Infomap(f'--two-level --silent --num-trials {n_trials}',
                 directed=False)

    # Add edges (upper triangle only to avoid duplicate links)
    cx = csr.tocoo()
    upper = cx.row < cx.col
    for s, t, w in zip(cx.row[upper], cx.col[upper], cx.data[upper]):
        im.add_link(int(s), int(t), float(w))

    im.run()

    n_nodes  = csr.shape[0]
    modules  = np.zeros(n_nodes, dtype=np.int32)
    for node in im.nodes:
        modules[node.node_id] = node.module_id

    return modules


def remove_small_communities(modules: np.ndarray,
                             min_size: int = MIN_COMMUNITY_SIZE) -> np.ndarray:
    """Set module id to 0 for all communities smaller than min_size."""
    out = modules.copy()
    ids, counts = np.unique(modules[modules > 0], return_counts=True)
    small = ids[counts < min_size]
    out[np.isin(modules, small)] = 0
    return out


# ---------------------------------------------------------------------------
# Consensus across densities
# ---------------------------------------------------------------------------

def consensus_assignment(all_modules: list[np.ndarray]) -> np.ndarray:
    """
    Collapse assignments across thresholds (sparse → dense).
    Each node gets the label from the sparsest threshold where it is assigned.
    Nodes unassigned at all thresholds stay 0.
    all_modules must be ordered sparse → dense (matches INFOMAP_DENSITIES).
    """
    n_nodes = all_modules[0].shape[0]
    consensus = np.zeros(n_nodes, dtype=np.int32)

    for modules in all_modules:   # sparse → dense
        update = (consensus == 0) & (modules > 0)
        consensus[update] = modules[update]

    return consensus


# ---------------------------------------------------------------------------
# Reference atlas helpers (used by 04b_relabel.py via importlib)
# ---------------------------------------------------------------------------

def load_reference_labels(ref_path: Path) -> np.ndarray:
    """
    Load a reference network atlas (.npz with 'labels' and 'hemi' arrays)
    and return labels aligned to the CIFTI cortical vertex ordering
    (L non-medial-wall vertices, then R non-medial-wall vertices).
    """
    d = np.load(ref_path)
    labels = d['labels']   # (n_all_vertices,)
    hemi   = d['hemi']     # (n_all_vertices,) — 'L' or 'R'

    roi_L = nib.load(FSLR_ROI['L']).darrays[0].data.astype(bool)  # (32492,)
    roi_R = nib.load(FSLR_ROI['R']).darrays[0].data.astype(bool)  # (32492,)

    labels_L = labels[hemi == 'L'][roi_L]
    labels_R = labels[hemi == 'R'][roi_R]
    return np.concatenate([labels_L, labels_R]).astype(np.int32)


def assign_network_labels(modules: np.ndarray,
                          ref_labels: np.ndarray) -> np.ndarray:
    """
    For each Infomap community assign the plurality reference network label
    (excluding reference label 0 = unassigned). Nodes with module=0 stay 0.
    """
    network_labels = np.zeros_like(modules)
    for mod_id in np.unique(modules[modules > 0]):
        mask = modules == mod_id
        votes = ref_labels[mask]
        votes = votes[votes > 0]
        if len(votes) == 0:
            continue
        vals, counts = np.unique(votes, return_counts=True)
        network_labels[mask] = vals[np.argmax(counts)]
    return network_labels


# ---------------------------------------------------------------------------
# Spatial patch filter (Gordon 2017: remove pieces < 30 mm², dilate neighbours)
# ---------------------------------------------------------------------------

def _build_surface_adj_and_areas(surf_path: Path):
    """Sparse CSR adjacency + per-vertex area (mm²) from a .surf.gii."""
    surf   = nib.load(surf_path)
    coords = surf.darrays[0].data          # (n_verts, 3)
    faces  = surf.darrays[1].data          # (n_faces, 3)
    n = coords.shape[0]
    i = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    j = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
    adj = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    v0, v1, v2 = coords[faces[:,0]], coords[faces[:,1]], coords[faces[:,2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vert_areas = np.zeros(n)
    np.add.at(vert_areas, faces[:,0], face_areas / 3)
    np.add.at(vert_areas, faces[:,1], face_areas / 3)
    np.add.at(vert_areas, faces[:,2], face_areas / 3)
    return adj, vert_areas


def _dilate_into_removed(full_map: np.ndarray, adj: sp.csr_matrix,
                          cortex_mask: np.ndarray) -> np.ndarray:
    """
    BFS dilation: fill cortex vertices with label==0 using the most-common
    nonzero label among their surface neighbours. Propagates inward one
    vertex at a time (as in Gordon 2017) until all reachable gaps are filled.
    """
    from collections import deque
    result = full_map.copy()
    queue: deque = deque()
    in_queue: set = set()
    for v in np.where(cortex_mask & (result == 0))[0]:
        nbs = adj.indices[adj.indptr[v]:adj.indptr[v+1]]
        if np.any(result[nbs] > 0):
            queue.append(v)
            in_queue.add(v)
    while queue:
        v = queue.popleft()
        in_queue.discard(v)
        if result[v] != 0:
            continue
        nbs = adj.indices[adj.indptr[v]:adj.indptr[v+1]]
        nz = result[nbs]
        nz = nz[nz > 0]
        if len(nz) == 0:
            continue
        vals, counts = np.unique(nz, return_counts=True)
        result[v] = vals[np.argmax(counts)]
        for nb in nbs:
            if cortex_mask[nb] and result[nb] == 0 and nb not in in_queue:
                queue.append(nb)
                in_queue.add(nb)
    return result


def _filter_hemi(full_map: np.ndarray, adj: sp.csr_matrix,
                  vert_areas: np.ndarray, cortex_mask: np.ndarray,
                  min_area_mm2: float) -> tuple:
    """Remove patches < min_area_mm2 then dilate neighbours inward."""
    from scipy.sparse.csgraph import connected_components
    result = full_map.copy()
    n_removed = 0
    for lbl in np.unique(result[result > 0]):
        verts = np.where(result == lbl)[0]
        sub_adj = adj[verts][:, verts]
        n_comp, comp_ids = connected_components(sub_adj, directed=False)
        for c in range(n_comp):
            patch = verts[comp_ids == c]
            if vert_areas[patch].sum() < min_area_mm2:
                result[patch] = 0
                n_removed += len(patch)
    result = _dilate_into_removed(result, adj, cortex_mask)
    return result, n_removed


def filter_small_patches(network_labels: np.ndarray,
                          valid_mask: np.ndarray,
                          min_area_mm2: float = MIN_PATCH_AREA_MM2) -> np.ndarray:
    """
    Remove contiguous surface patches < min_area_mm2 and fill by dilation
    (Gordon et al. 2017: 30 mm² threshold on fsLR 32k template surface).

    Parameters
    ----------
    network_labels : (n_valid,) int32 — reference network labels for valid vertices
    valid_mask     : (n_cortex,) bool — which of the ~59k cortical vertices are valid
    min_area_mm2   : area threshold in mm² (default 30)

    Returns
    -------
    Filtered network_labels in the same (n_valid,) space.
    """
    roi_L = nib.load(FSLR_ROI['L']).darrays[0].data.astype(bool)   # (32492,)
    roi_R = nib.load(FSLR_ROI['R']).darrays[0].data.astype(bool)   # (32492,)
    n_cortex_L = roi_L.sum()

    adj_L, areas_L = _build_surface_adj_and_areas(FSLR_MIDTHICK['L'])
    adj_R, areas_R = _build_surface_adj_and_areas(FSLR_MIDTHICK['R'])

    # Expand valid-mask labels → full cortex space
    cortex_labels = np.zeros(n_cortex_L + roi_R.sum(), dtype=np.int32)
    cortex_labels[valid_mask] = network_labels

    # Expand cortex space → full 32k hemisphere arrays
    full_L = np.zeros(32492, dtype=np.int32)
    full_L[roi_L] = cortex_labels[:n_cortex_L]
    full_R = np.zeros(32492, dtype=np.int32)
    full_R[roi_R] = cortex_labels[n_cortex_L:]

    full_L, rm_L = _filter_hemi(full_L, adj_L, areas_L, roi_L, min_area_mm2)
    full_R, rm_R = _filter_hemi(full_R, adj_R, areas_R, roi_R, min_area_mm2)

    if rm_L + rm_R > 0:
        print(f'  Patch filter: {rm_L + rm_R} vertices removed and filled '
              f'({rm_L} L, {rm_R} R)')

    # Collapse back → valid_mask space
    cortex_new = np.concatenate([full_L[roi_L], full_R[roi_R]])
    return cortex_new[valid_mask].astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f'{s/60:.1f} min' if s >= 60 else f'{s:.1f} s'


def main(subject: str, single_density: float | None = None) -> None:
    t_total = time.perf_counter()
    cm_dir  = OUTPUT_ROOT / subject / 'cm'
    out_dir = OUTPUT_ROOT / subject / 'networks'
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f'{subject}_{SESSION}_space-fsLR32k'
    densities = [single_density] if single_density else INFOMAP_DENSITIES
    all_modules = []

    for d in densities:
        d_str   = f'{d:.3f}'.replace('0.', '')
        cm_path = cm_dir / f'{stem}_density-{d_str}_cm.npz'
        if not cm_path.exists():
            print(f'  [skip] {cm_path.name} not found — run 03_vertex_cm.py first')
            continue

        out_path = out_dir / f'{stem}_density-{d_str}_communities.npz'
        if out_path.exists():
            print(f'  [skip] {out_path.name} already exists — loading for consensus')
            all_modules.append(np.load(out_path)['modules'])
            continue

        print(f'[{subject}] Running Infomap at density={d:.3f} ...')
        t = time.perf_counter()
        csr     = sp.load_npz(cm_path)
        n_nodes = csr.shape[0]
        print(f'  Graph: {n_nodes} nodes, {csr.nnz//2:,} edges')

        modules = run_infomap(csr)
        modules = remove_small_communities(modules)

        n_assigned = (modules > 0).sum()
        n_nets     = len(np.unique(modules[modules > 0]))
        print(f'  → {n_nets} communities, {100*n_assigned/n_nodes:.1f}% nodes assigned')

        np.savez(out_path, modules=modules, density=np.array([d]))
        print(f'  Saved → {out_path.name}  [{_elapsed(t)}]')
        all_modules.append(modules)

    if len(all_modules) > 1:
        print(f'[{subject}] Computing consensus across {len(all_modules)} thresholds ...')
        consensus = consensus_assignment(all_modules)
        n_nets = len(np.unique(consensus[consensus > 0]))
        print(f'  → {n_nets} consensus communities')

        out_path = out_dir / f'{stem}_consensus_communities.npz'
        np.savez(out_path, modules=consensus)
        print(f'  Saved → {out_path.name}')

    print(f'\n[{subject}] Total time: {_elapsed(t_total)}')


def parse_subject(s: str) -> str:
    return f'sub-{int(s.removeprefix("sub-")):02d}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='subject id: 1, 01, or sub-01')
    parser.add_argument('--density', type=float, default=None,
                        help='Run single density only (e.g. 0.005)')
    args = parser.parse_args()
    main(parse_subject(args.subject), args.density)
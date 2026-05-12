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
from config import OUTPUT_ROOT, SESSION, INFOMAP_DENSITIES, FSLR_ROI

MIN_COMMUNITY_SIZE = 400   # communities smaller than this are marked unassigned


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
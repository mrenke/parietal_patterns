#!/usr/bin/env python3
"""
Step 4 — Infomap network detection (Gordon 2017 protocol).

Runs the Infomap community detection algorithm on the thresholded
vertex-wise correlation matrix produced by 03_vertex_cm.py.

Gordon 2017 protocol:
  - Two-level Infomap on undirected weighted graph
  - Thresholds from 0.3% to 5% density
  - Small communities (< 400 nodes) removed
  - Network identities assigned by spatial overlap with a reference atlas
  - Consensus assignment by collapsing across density thresholds

Saved files (OUTPUT_ROOT/sub-XX/networks/):
  Without --ref:
    sub-XX_ses-1_space-fsLR32k_density-{d}_communities.npz
    sub-XX_ses-1_space-fsLR32k_consensus_communities.npz
  With --ref:
    sub-XX_ses-1_space-fsLR32k_density-{d}_ref-{name}_communities.npz
    sub-XX_ses-1_space-fsLR32k_consensus_ref-{name}_communities.npz
    Each .npz adds 'network_labels': plurality-vote reference label per node.

Reference atlas format (.npz):
  labels : (n_all_vertices,) int32  — network label per fsLR 32k vertex
  hemi   : (n_all_vertices,) str    — 'L' or 'R' per vertex
  (medial-wall vertices are filtered out using the HCP atlas ROI masks)

Usage:
  python 04_infomap.py sub-01
  python 04_infomap.py sub-01 --density 0.005
  python 04_infomap.py sub-01 --ref /path/to/atlas.npz
  python 04_infomap.py sub-01 --ref /path/to/atlas.npz --ref-name myAtlas
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
# Main
# ---------------------------------------------------------------------------
# Reference atlas alignment and network label assignment
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


def main(subject: str, single_density: float | None = None,
         ref_path: Path | None = None, ref_name: str | None = None) -> None:
    t_total = time.perf_counter()
    cm_dir  = OUTPUT_ROOT / subject / 'cm'
    out_dir = OUTPUT_ROOT / subject / 'networks'
    out_dir.mkdir(parents=True, exist_ok=True)

    meta       = np.load(cm_dir / f'{subject}_{SESSION}_space-fsLR32k_cm_meta.npz')
    valid_mask = meta['valid_mask']   # (n_cifti_nodes,) bool

    # Load and align reference labels if provided
    ref_labels = None
    ref_tag    = ''
    if ref_path is not None:
        print(f'[{subject}] Loading reference atlas: {ref_path.name}')
        ref_labels_cifti = load_reference_labels(ref_path)
        ref_labels = ref_labels_cifti[valid_mask]
        ref_tag = f'_ref-{ref_name or ref_path.name.split("_space-")[0]}'
        print(f'  {len(np.unique(ref_labels[ref_labels > 0]))} reference networks, '
              f'{ref_labels.shape[0]} nodes')

    stem = f'{subject}_{SESSION}_space-fsLR32k'
    densities = [single_density] if single_density else INFOMAP_DENSITIES
    all_modules = []

    for d in densities:
        d_str   = f'{d:.3f}'.replace('0.', '')
        cm_path = cm_dir / f'{stem}_density-{d_str}_cm.npz'
        if not cm_path.exists():
            print(f'  [skip] {cm_path.name} not found — run 03_vertex_cm.py first')
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
        print(f'  → {n_nets} communities, '
              f'{100*n_assigned/n_nodes:.1f}% nodes assigned')

        save_kwargs = dict(modules=modules, density=np.array([d]))
        if ref_labels is not None:
            net_labels = assign_network_labels(modules, ref_labels)
            save_kwargs['network_labels'] = net_labels
            n_nets_ref = len(np.unique(net_labels[net_labels > 0]))
            print(f'  → {n_nets_ref} reference networks assigned')

        out_path = out_dir / f'{stem}_density-{d_str}{ref_tag}_communities.npz'
        np.savez(out_path, **save_kwargs)
        print(f'  Saved → {out_path.name}  [{_elapsed(t)}]')
        all_modules.append(modules)

    # Consensus (only if multiple densities)
    if len(all_modules) > 1:
        print(f'[{subject}] Computing consensus across {len(all_modules)} thresholds ...')
        consensus = consensus_assignment(all_modules)
        n_nets = len(np.unique(consensus[consensus > 0]))
        print(f'  → {n_nets} consensus communities')

        save_kwargs = dict(modules=consensus)
        if ref_labels is not None:
            net_labels = assign_network_labels(consensus, ref_labels)
            save_kwargs['network_labels'] = net_labels

        out_path = out_dir / f'{stem}_consensus{ref_tag}_communities.npz'
        np.savez(out_path, **save_kwargs)
        print(f'  Saved → {out_path.name}')

    print(f'\n[{subject}] Total time: {_elapsed(t_total)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='e.g. sub-01')
    parser.add_argument('--density', type=float, default=None,
                        help='Run single density only (e.g. 0.005)')
    parser.add_argument('--ref', type=Path, default=None,
                        help='Reference atlas .npz for network label assignment')
    parser.add_argument('--ref-name', type=str, default=None,
                        help='Short name for the reference atlas (used in output filenames)')
    args = parser.parse_args()
    main(args.subject, args.density, args.ref, args.ref_name)

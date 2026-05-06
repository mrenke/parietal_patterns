#!/usr/bin/env python3
"""
Step 4b — Re-assign network labels using a different reference atlas.

Skips Infomap entirely. Loads existing per-density community files
(produced by 04_infomap.py), applies a new reference atlas, and saves
new output files with the reference name in the filename.

Requires 04_infomap.py to have been run first (modules must exist on disk).

Usage:
  python 04b_relabel.py sub-01 --ref /path/to/atlas.npz
  python 04b_relabel.py sub-01 --ref /path/to/atlas.npz --ref-name myAtlas
  python 04b_relabel.py sub-01 --ref /path/to/atlas.npz --density 0.005
"""
import sys
import argparse
import importlib.util
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_ROOT, SESSION, INFOMAP_DENSITIES

# Import functions from 04_infomap.py (digit prefix prevents normal import)
_spec = importlib.util.spec_from_file_location(
    'infomap', Path(__file__).parent / '04_infomap.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_reference_labels = _mod.load_reference_labels
assign_network_labels = _mod.assign_network_labels
consensus_assignment  = _mod.consensus_assignment


def find_modules_file(out_dir: Path, stem: str, d_str: str) -> Path | None:
    """Return existing community file for this density (no-ref preferred)."""
    exact = out_dir / f'{stem}_density-{d_str}_communities.npz'
    if exact.exists():
        return exact
    matches = sorted(out_dir.glob(f'{stem}_density-{d_str}_*communities.npz'))
    return matches[0] if matches else None


def main(subject: str, ref_path: Path, ref_name: str | None,
         single_density: float | None) -> None:
    cm_dir  = OUTPUT_ROOT / subject / 'cm'
    out_dir = OUTPUT_ROOT / subject / 'networks'

    # Load and align reference labels
    print(f'[{subject}] Loading reference atlas: {ref_path.name}')
    meta       = np.load(cm_dir / f'{subject}_{SESSION}_space-fsLR32k_cm_meta.npz')
    valid_mask = meta['valid_mask']
    ref_labels_cifti = load_reference_labels(ref_path)
    ref_labels = ref_labels_cifti[valid_mask]
    ref_tag    = f'_ref-{ref_name or ref_path.name.split("_space-")[0]}'
    print(f'  {len(np.unique(ref_labels[ref_labels > 0]))} reference networks, '
          f'{ref_labels.shape[0]} nodes')

    stem = f'{subject}_{SESSION}_space-fsLR32k'
    densities = [single_density] if single_density else INFOMAP_DENSITIES
    all_modules = []

    for d in densities:
        d_str    = f'{d:.3f}'.replace('0.', '')
        src_path = find_modules_file(out_dir, stem, d_str)
        if src_path is None:
            print(f'  [skip] no modules found for density={d:.3f} '
                  f'— run 04_infomap.py first')
            continue

        print(f'[{subject}] Loading density={d:.3f} from {src_path.name} ...')
        modules    = np.load(src_path)['modules']
        net_labels = assign_network_labels(modules, ref_labels)
        all_modules.append(net_labels)

    # Consensus on reference-aligned labels (comparable across thresholds)
    print(f'[{subject}] Computing consensus across {len(all_modules)} thresholds ...')
    consensus  = consensus_assignment(all_modules)
    net_labels = consensus
    n_nets     = len(np.unique(net_labels[net_labels > 0]))
    print(f'  → {n_nets} reference networks in consensus')

    out_path = out_dir / f'{stem}_consensus{ref_tag}_communities.npz'
    np.savez(out_path, modules=consensus, network_labels=net_labels)
    print(f'  Saved → {out_path.name}')


def parse_subject(s: str) -> str:
    return f'sub-{int(s.removeprefix("sub-")):02d}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='subject id: 1, 01, or sub-01')
    parser.add_argument('--ref', type=Path, required=True,
                        help='Reference atlas .npz for network label assignment')
    parser.add_argument('--ref-name', type=str, default=None,
                        help='Short name for the reference (used in output filenames)')
    parser.add_argument('--density', type=float, default=None,
                        help='Relabel single density only (e.g. 0.005)')
    args = parser.parse_args()
    main(parse_subject(args.subject), args.ref, args.ref_name, args.density)

#!/usr/bin/env python3
"""
Build a reference atlas for 04b (ATLAS_DIR/<name>_space-fsLR_den-32k_cortex.npz,
format: labels + hemi over all 2 × 32,492 fsLR vertices, medial wall = 0).

Two sources:
  --ca19                  original Cole-Anticevic network partition (Ji et al. 2019,
                          CAB-NP) from hcp_utils (91k grayordinates → cortex part)
  --subject S --ref R     a (group-average) PFM result: the patch-filtered 04b
                          network_labels of sub-S labelled with reference R

Names follow <label set><task><data source> (see config.py), e.g.
  python make_ref_atlas.py --ca19                                       # → ca19
  PFM_DATASET=asd python make_ref_atlas.py --subject average --ref gordon17 \\
      --name gordon17chaseASDC

Provenance of every atlas is appended to ATLAS_DIR/README.md.
"""
import argparse
import sys
from datetime import date
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (ATLAS_DIR, BASE_OUTPUT_ROOT, SESSION, FSLR_ROI,
                    atlas_path, label_set)


def expand_cortex(cortex_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(59,412,) cortex labels (L then R, no medial wall) → (64,984,) labels + hemi."""
    roi = {h: nib.load(FSLR_ROI[h]).darrays[0].data.astype(bool) for h in 'LR'}
    n_L = roi['L'].sum()
    if len(cortex_labels) != n_L + roi['R'].sum():
        raise ValueError(f'expected {n_L + roi["R"].sum()} cortex labels, got {len(cortex_labels)}')
    full = {h: np.zeros(len(roi[h]), dtype=np.int32) for h in 'LR'}
    full['L'][roi['L']] = cortex_labels[:n_L]
    full['R'][roi['R']] = cortex_labels[n_L:]
    labels = np.concatenate([full['L'], full['R']])
    hemi   = np.repeat(['L', 'R'], [len(full['L']), len(full['R'])])
    return labels, hemi


def from_ca19() -> np.ndarray:
    import hcp_utils as hcp
    cortex = hcp.ca_network.map_all[:hcp.struct.cortex.stop]   # cortex L + R
    return cortex.astype(np.int32)


def from_subject(subject: str, ref: str) -> np.ndarray:
    stem = f'{subject}_{SESSION}_space-fsLR32k'
    valid = np.load(BASE_OUTPUT_ROOT / subject / 'cm' / f'{stem}_cm_meta.npz')['valid_mask']
    net   = np.load(BASE_OUTPUT_ROOT / subject / 'networks' /
                    f'{stem}_consensus_ref-{ref}_communities.npz')['network_labels']
    cortex = np.zeros(len(valid), dtype=np.int32)
    cortex[valid] = net
    return cortex


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--ca19', action='store_true')
    src.add_argument('--subject', help="e.g. 'average' or 'average-ASD'")
    parser.add_argument('--ref', help='with --subject: reference its 04b labels used')
    parser.add_argument('--name', help="atlas name (default 'ca19' with --ca19)")
    parser.add_argument('--note', default='', help='extra provenance text for the README')
    args = parser.parse_args()

    if args.ca19:
        name, cortex = args.name or 'ca19', from_ca19()
        source = 'Cole-Anticevic network partition (Ji et al. 2019, NeuroImage), hcp_utils ca_network'
    else:
        if not (args.ref and args.name):
            parser.error('--subject needs --ref and --name')
        subject = f'sub-{args.subject.removeprefix("sub-")}'
        name, cortex = args.name, from_subject(subject, args.ref)
        source = (f'{BASE_OUTPUT_ROOT}/{subject}: 04b network_labels '
                  f'(patch-filtered consensus), ref {args.ref}')
    if label_set(name) != (label_set(args.ref) if args.ref else 'ca19'):
        sys.exit(f'Name {name!r} must start with the label set of its labels')

    labels, hemi = expand_cortex(cortex)
    out = atlas_path(name)
    if out.exists():
        sys.exit(f'{out} exists — remove it first to rebuild')
    np.savez(out, labels=labels, hemi=hemi)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    print(f'Saved {out}\n  {len(ids)} networks: ' +
          ', '.join(f'{i}:{c}' for i, c in zip(ids, counts)))

    with open(ATLAS_DIR / 'README.md', 'a') as f:
        f.write(f'- `{name}` ({date.today()}): {source}. {args.note}\n')

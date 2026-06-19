#!/usr/bin/env python3
"""
Step 5 — Plot individual network maps and save to shared plot folder.

Generates one 2×2 figure (L/R × lateral/medial) per reference atlas:
  - gordon17
  - caNets_DDnr

Saved to:
  /mnt_AdaBD_largefiles/.../plots_and_ims/nets_PFM/
    sub-XX_ref-gordon17_networks.png
    sub-XX_ref-caNets_DDnr_networks.png

Usage:
  python 05_plot.py sub-01
  python 05_plot.py 1
"""
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving
import matplotlib.pyplot as plt
from pathlib import Path
from neuromaps.datasets import fetch_atlas

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_ROOT, HCP_ATLASES, SESSION
from utils_01 import get_gordon17_cmap, get_CANets_cmap


PLOT_DIR = Path('/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk'
                '/plots_and_ims/nets_PFM')

REFS = {
    'gordon17':   {'vmax': 17, 'cmap': None},   # filled below
    'caNets_DDnr': {'vmax': 12, 'cmap': None},
}


# ---------------------------------------------------------------------------
# Label prep (from notebook)
# ---------------------------------------------------------------------------

def prep_labels_for_visualization(labels: np.ndarray,
                                   valid_mask: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Expand (n_valid_nodes,) → two (32492,) arrays, one per hemisphere."""
    roi_L = nib.load(HCP_ATLASES / 'L.atlasroi.32k_fs_LR.shape.gii').darrays[0].data.astype(bool)
    roi_R = nib.load(HCP_ATLASES / 'R.atlasroi.32k_fs_LR.shape.gii').darrays[0].data.astype(bool)
    n_L_cortex = roi_L.sum()

    cifti_labels = np.zeros(len(valid_mask), dtype=np.int32)
    cifti_labels[valid_mask] = labels

    lh = np.zeros(len(roi_L), dtype=np.int32)
    rh = np.zeros(len(roi_R), dtype=np.int32)
    lh[roi_L] = cifti_labels[:n_L_cortex]
    rh[roi_R] = cifti_labels[n_L_cortex:]
    return lh, rh


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_networks(subject: str, ref_name: str, lh_labels: np.ndarray,
                  rh_labels: np.ndarray, cmap, vmax: int,
                  fslr: dict) -> plt.Figure:
    from nilearn import plotting

    lh_surf  = str(fslr['inflated'].L)
    rh_surf  = str(fslr['inflated'].R)
    lh_sulc  = str(fslr['sulc'].L)
    rh_sulc  = str(fslr['sulc'].R)

    panels = [
        (lh_surf, lh_labels, 'left',  'lateral', lh_sulc),
        (lh_surf, lh_labels, 'left',  'medial',  lh_sulc),
        (rh_surf, rh_labels, 'right', 'lateral', rh_sulc),
        (rh_surf, rh_labels, 'right', 'medial',  rh_sulc),
    ]

    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'},
                             figsize=(12, 8))
    for ax, (surf, data, hemi, view, sulc) in zip(axes.flat, panels):
        plotting.plot_surf(
            surf, data,
            hemi=hemi, view=view,
            cmap=cmap,
            vmin=0, vmax=vmax,
            avg_method='median',
            darkness=1.0,
            colorbar=False,
            bg_map=sulc,
            bg_on_data=True,
            axes=ax,
        )
        ax.set_title(f'{hemi} — {view}', fontsize=10)

    fig.suptitle(f'{subject}  |  ref: {ref_name}', fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subject: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load surfaces once
    fslr = fetch_atlas('fsLR', '32k')

    # Load cmeta (valid_mask)
    meta_path = (OUTPUT_ROOT / subject / 'cm' /
                 f'{subject}_{SESSION}_space-fsLR32k_cm_meta.npz')
    if not meta_path.exists():
        raise FileNotFoundError(f'Run 03_vertex_cm.py first: {meta_path}')
    valid_mask = np.load(meta_path)['valid_mask']

    # Colormaps
    cmap_gordon, _ = get_gordon17_cmap()
    cmap_ca        = get_CANets_cmap()
    REFS['gordon17']['cmap']    = cmap_gordon
    REFS['caNets_DDnr']['cmap'] = cmap_ca

    for ref_name, cfg in REFS.items():
        net_path = (OUTPUT_ROOT / subject / 'networks' /
                    f'{subject}_{SESSION}_space-fsLR32k'
                    f'_consensus_ref-{ref_name}_communities.npz')
        if not net_path.exists():
            print(f'  [skip] {net_path.name} not found')
            continue

        print(f'[{subject}] Plotting ref={ref_name} ...')
        net_labels = np.load(net_path)['modules']  # ['network_labels']
        lh, rh     = prep_labels_for_visualization(net_labels, valid_mask)

        fig = plot_networks(subject, ref_name, lh, rh,
                            cfg['cmap'], cfg['vmax'], fslr)

        out = PLOT_DIR / f'{subject}_ref-{ref_name}_networks.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {out}')


def parse_subject(s: str) -> str:
    return f'sub-{int(s.removeprefix("sub-")):02d}'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python 05_plot.py <subject>  # e.g. 1, 01, or sub-01')
        sys.exit(1)
    main(parse_subject(sys.argv[1]))

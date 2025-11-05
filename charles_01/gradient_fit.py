from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
import numpy as np
import nibabel as nib
from nilearn import datasets
import os.path as op
import os
from nilearn import signal
import pandas as pd
from scipy.sparse.csgraph import connected_components

# import of get_basic_mask
try:
    # direkter Import, wenn gradients_noHalo als Paket erkannt wird
    from gradients_noHalo.utils import get_basic_mask
except ModuleNotFoundError:
    import sys
    # Füge das Repo-Root (Ordner eine Ebene höher als charles_01) hinzu
    repo_root = op.abspath(op.join(op.dirname(__file__), ".."))
    sys.path.append(repo_root)
    from gradients_noHalo.utils import get_basic_mask

import glob
import re


# not needed specifications (for filename only/logging)
frames_per_run_thresh=104
scrub_thresh=0.3
lower_bpf=0.01
upper_bpf=0.08

def main(sub, confspec, bids_folder, ses=1, task='magjudge', specification='',
        scrubbing=True,
        run_FD_filter=True, 
        bp_filtering=True,
        space='fsaverage5',
        ref_dir='gradients.tryNoHalo', 
        n_components=10):
    
    """ 
    Berechnet Gradienten für einen Subjekt, basierend auf den Konnektivitätsmatrizen in BIDS-Ordner. 
    Optional werden Prokrustes-Aligment auf Referenz-Gradienten durchgeführt.
    """
            
    # Subjekt ID zweistellig
    sub = f'{int(sub):02d}'

    # confspec string (nur für Dateinamen)
    #confspec += f'scrub{str(scrub_thresh)[2]}'  if scrubbing else confspec
    #confspec += 'BPfilter' if bp_filtering else confspec
    #confspec += f'runFD{str(frames_per_run_thresh)}' if run_FD_filter else confspec
    #key = f'.{confspec}'

    # robustere confspec-Zusammensetzung (kein Verdoppeln mehr)
    if scrubbing:
        confspec += f'scrub{str(scrub_thresh)[2]}'
    if bp_filtering:
        confspec += 'BPfilter'
    if run_FD_filter:
        confspec += f'runFD{str(frames_per_run_thresh)}'
    key = f'.{confspec}'

    target_dir = op.join(bids_folder, 'derivatives', f'gradients{key}', f'sub-{sub}')
    os.makedirs(target_dir, exist_ok=True) # create target dir if not exists

    # CM-Datei finden und laden
    patterns = [
        # Original-Schema mit tryNoHalo & *runs_CM-unfiltered.npy
        op.join(bids_folder,'derivatives','correlation_matrices.tryNoHalo',
                f'sub-{sub}_ses-{ses}_task-{task}_space-{space}_confspec-{confspec}-*runs_CM-unfiltered.npy'),
        # Dein CM-Schema aus dem Cleaning-Skript (_CM.npy)
        op.join(bids_folder,'derivatives','correlation_matrices',
                f'sub-{sub}_ses-{ses}_task-{task}_space-{space}_confspec-{confspec}_CM.npy'),
    ]
    files = []
    for p in patterns:
        files += glob.glob(p)
    files = sorted(files)
    if not files:
        raise FileNotFoundError("Keine CM-Datei gefunden für Muster:\n  " + "\n  ".join(patterns))
    sub_file = files[-1]

    m = re.search(r'runFD\d+-(\d+)runs_CM-unfiltered\.npy', op.basename(sub_file))
    n_runs = m.group(1) if m else 'NA'
    print(f'Loading {sub_file}\nsub {sub} had {n_runs} sufficient runs')

    cm = np.load(sub_file)
    # Für Graph-Operationen: NaNs -> 0, Diagonale = 0
    if np.isnan(cm).any():
        cm = np.nan_to_num(cm, copy=False)
    np.fill_diagonal(cm, 0.0)

    # gröste zusammenhängende Komponente bestimmen und Maske speichern/laden
    cc_mask_file = op.join(target_dir, f'sub-{sub}_cc-mask_space-{space}.npy')
    if not op.exists(cc_mask_file):
        n_comp, labels = connected_components(cm)
        largest = np.bincount(labels).argmax()
        mask_cc = labels == largest
        np.save(cc_mask_file, mask_cc)
        print(f'connected components derived ({n_comp} comps) & mask saved')
    mask_cc = np.load(cc_mask_file)
    
    # Basis-Maske laden und labeling (ohne Parcels) für Mapping
    mask, labeling_noParcel = get_basic_mask()  # liefert (bool mask, labeling)
    # Konsistenzcheck: Länge der CC-Maske = Anzahl True in Basis-Maske
    if mask.sum() != mask_cc.size:
        raise ValueError(f'Mask mismatch: mask.sum()={mask.sum()} vs mask_cc.size={mask_cc.size}')
    mask = mask.copy()
    mask[mask] = mask_cc   # kombiniere Basis- & CC-Maske

    cm_filtered = cm[mask_cc, :][:, mask_cc]
    print('connectivity matrix loaded and filtered with cc_mask')

    # reference gradients
    g_ref_fn = op.join(bids_folder, 'derivatives', ref_dir, 'sub-All',
                       f'sub-All_gradients_space-{space}_confspec-{confspec}.npy')
    use_reference = op.exists(g_ref_fn)
    if use_reference:
        g_ref = np.load(g_ref_fn)          # shape: (n_comp_ref, n_vertices_total)
        g_ref_fil = g_ref[:, mask].T       # -> (N_kept_vertices, n_comp_ref)
        align = 'procrustes'
        reference = g_ref_fil
        print(f'Loaded reference gradients: {g_ref_fn}')
    else:
        align = None
        reference = None
        print(f'No reference at {g_ref_fn} → fitting without alignment.')

    # gradient fitting
    gm = GradientMaps(n_components=n_components, alignment=align)
    gm.fit(cm_filtered, reference=reference)

    # save results
    np.save(op.join(target_dir, f'sub-{sub}_lambdas_space-{space}_n{n_components}{specification}.npy'),
            gm.lambdas_) # save all together
    
    # Unaligned → volle Fläche
    gm_unaligned = gm.gradients_.T  # (n_components, N_kept)
    grad = [map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan) for g in gm_unaligned]
    np.save(op.join(target_dir, f'sub-{sub}_gradients_space-{space}_n{n_components}{specification}.npy'),
            grad)

    # Aligned (falls vorhanden) → volle Fläche
    if getattr(gm, 'aligned_', None) is not None:
        gm_aligned = gm.aligned_.T
        grad_al = [map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan) for g in gm_aligned]
        np.save(op.join(target_dir, f'sub-{sub}_g-aligned_space-{space}_n{n_components}{specification}.npy'),
                grad_al)

    print(f'finished sub-{sub} ses-{ses} task-{task} - {confspec}: gradients saved in {target_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--confspec', default='32P')
    parser.add_argument('--bids_folder', default='/mnt_01/ds-dnumrisk')
    parser.add_argument('--ses', type=int, default=1)
    parser.add_argument('--task', default='magjudge')
    parser.add_argument('--space', default='fsaverage5')
    parser.add_argument('--specification', default='')
    parser.add_argument('--ref_dir', default='gradients.tryNoHalo')
    parser.add_argument('--n_components', type=int, default=10)

    # Flags analog Vorlage (nur Naming):
    parser.add_argument('--scrubbing', dest='scrubbing', action='store_true')
    parser.add_argument('--no-scrubbing', dest='scrubbing', action='store_false')
    parser.set_defaults(scrubbing=True)

    parser.add_argument('--run_FD_filter', dest='run_FD_filter', action='store_true')
    parser.add_argument('--no-run_FD_filter', dest='run_FD_filter', action='store_false')
    parser.set_defaults(run_FD_filter=True)

    parser.add_argument('--bp_filtering', dest='bp_filtering', action='store_true')
    parser.add_argument('--no-bp_filtering', dest='bp_filtering', action='store_false')
    parser.set_defaults(bp_filtering=True)

    args = parser.parse_args()
    main(args.subject, args.confspec, args.bids_folder,
         ses=args.ses, task=args.task, specification=args.specification,
         scrubbing=args.scrubbing, run_FD_filter=args.run_FD_filter, bp_filtering=args.bp_filtering,
         space=args.space, ref_dir=args.ref_dir, n_components=args.n_components)

import argparse
import glob
import re
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
from utils import get_basic_mask
import time

mask, labeling_noParcel = get_basic_mask()

def main(subject, bids_folder, ses=1, task='magjudge', confspec='36P',
        kernel = None, #'cosine',
        ztransf = False,
        alignRef = '-tanH'):
    start_time = time.time()
    sub = f'{int(subject):02d}'
    key = f'.tryParams.{confspec}'
    target_dir = op.join(bids_folder, 'derivatives', f'gradients{key}', f'sub-{sub}')
    specification = f'kernel-{kernel}_ztransf-{ztransf}'
    print(f'Starting {sub}; specification: {specification}, alignRef: {alignRef}')

    # get regular CM 
    cm_confspec = f'{confspec}scrub3BPfilterrunFD104'
    os.makedirs(target_dir, exist_ok=True)
    sub_file_pattern = op.join(bids_folder,'derivatives','correlation_matrices.tryNoHalo', f'sub-{sub}_ses-{ses}_task-{task}_confspec-{cm_confspec}-*runs_CM-unfiltered.npy')
    sub_file = glob.glob(sub_file_pattern)[0]
    n_runs = (re.search(r'runFD104-(\d+)runs_CM-unfiltered.npy', sub_file)).group(1)
    print(f'Loading {sub_file} - \n sub {sub} had {n_runs} sufficient runs')
    cm = np.load(sub_file)

    if ztransf:
        cm = np.arctanh(cm) # "....normalized the correlation coefficients using Fisher’s z-transformation -  
        cm[np.isinf(cm)] = 0
    # statistische Methode, die den Pearson-Korrelationskoeffizienten (\(r\)) in eine normalverteilte Variable (\(z^{\prime }\)) umwandel = its inverse hyperbolic tangent (artanh).

    if kernel == None: # also needed for cosine I think
        print('Applying CC-mask for kernel=None')
        cc_mask_file = op.join(op.join(bids_folder, 'derivatives', f'gradients.{cm_confspec}', f'sub-{sub}'), f'sub-{sub}_cc-mask_space-fsaverag5.npy')
        mask_cc = np.load(cc_mask_file)
        mask[mask == True] = mask_cc
        cm = cm[mask_cc, :][:, mask_cc]

    # reference gradients
    #ref_grad = op.join(bids_folder, 'derivatives', f'gradients{key}', f'sub-All', f'sub-All_ses-{ses}_task-{task}_gradients_confspec-{confspec}_alignRef{alignRef}.npy')
    #ref_grad = op.join(bids_folder, 'derivatives', f'gradients{key}',f'sub-All_ses-1_task-magjudge_gradients_kernel-None_ztransf-False_avMethod{alignRef}.npy')
    ref_grad = op.join(bids_folder, 'derivatives', f'gradients.tryParams.36P','sub-All', f'sub-All_gradients_{specification}_avMethod{alignRef}.npy')
    grad_ref = np.load(ref_grad)
    grad_ref_fil = grad_ref[:,mask].T  # only use nodes in mask
    
    # Fit gradients
    n_components = 10
    gm = GradientMaps(n_components=n_components, alignment='procrustes', kernel=kernel, approach='dm', random_state=0)
    gm.fit(cm, reference=grad_ref_fil)

    # save results
    np.save(op.join(target_dir,f'sub-{sub}_lambdas_{specification}.npy'), gm.lambdas_) 

    gm_= gm.gradients_.T 
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_gradients_{specification}.npy'), grad) 
    gm_ = gm.aligned_.T
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    fn = op.join(target_dir,f'sub-{sub}_g-aligned{alignRef}_{specification}.npy')
    np.save(fn, grad) 

    elapsed_time = time.time() - start_time
    print(f'Finished sub - {sub} in {elapsed_time/60:.2f} minutes. \n Saved aligned gradients to {fn}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk', help='Path to BIDS folder')

    parser.add_argument('--kernel', type=str, help='Kernel type for GradientMaps') # if kernel none, CC-mask filtering is needed probably!
    parser.add_argument('--ztransf', action='store_true', help='Apply Fisher z-transformation to connectivity matrix')
    parser.add_argument('--alignRef', type=str, default='-tanH', help='Reference alignment method')

    args = parser.parse_args()

    main(args.subject, args.bids_folder, 
        kernel=args.kernel, ztransf=args.ztransf, alignRef=args.alignRef)

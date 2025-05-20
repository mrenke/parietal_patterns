import argparse
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
from utils import get_basic_mask

import glob
import re

# not needed specifications
frames_per_run_thresh=104
scrub_thresh=0.3
lower_bpf=0.01
upper_bpf=0.08

def main(sub, confspec, bids_folder, ses=1, task='magjudge', specification='',
        scrubbing=True,
        run_FD_filter=True, 
        bp_filtering=True,):  
            
    sub = f'{int(sub):02d}'

    confspec += f'scrub{str(scrub_thresh)[2]}'  if scrubbing else confspec
    confspec += 'BPfilter' if bp_filtering else confspec
    confspec += f'runFD{str(frames_per_run_thresh)}' if run_FD_filter else confspec
    key = f'.{confspec}'

    target_dir = op.join(bids_folder, 'derivatives', f'gradients{key}', f'sub-{sub}')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    #cm_fn = op.join(bids_folder, 'derivatives', 'correlation_matrices.tryNoHalo', f'sub-{sub}_ses-{ses}_task-magjudge_confspec-{confspec}_CM-unfiltered.npy')
    sub_file_pattern = op.join(bids_folder,'derivatives','correlation_matrices.tryNoHalo', f'sub-{sub}_ses-{ses}_task-{task}_confspec-{confspec}-*runs_CM-unfiltered.npy')
    sub_file = glob.glob(sub_file_pattern)[0]
    n_runs = (re.search(r'runFD104-(\d+)runs_CM-unfiltered.npy', sub_file)).group(1)
    print(f'Loading {sub_file} - \n sub {sub} had {n_runs} sufficient runs')
    cm = np.load(sub_file)

    # dont take old cc mask, but new one
    cc_mask_file = op.join(target_dir,f'sub-{sub}_cc-mask_space-fsaverag5.npy')
    if (os.path.exists(cc_mask_file) == False):
        cc = connected_components(cm)
        mask_cc = cc[1] == 0 # all nodes in 0 belong to the largest connected component, check #-components in cc[0]
        np.save(cc_mask_file, mask_cc) # save all together
        print('connected components derived & mask saved')  
    #cc_mask_file = op.join(bids_folder_old,'derivatives', f'gradients', f'sub-{sub}', f'sub-{sub}_cc-mask_space-fsaverag5.npy') # from before!
    mask_cc = np.load(cc_mask_file)
    
    mask, labeling_noParcel = get_basic_mask()
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask
    cm_filtered = cm[mask_cc, :][:, mask_cc]
    print('connectivty matrix loaded and filtered with cc_mask')  

    # reference gradients
    g_ref_fn = op.join(bids_folder,'derivatives','gradients.tryNoHalo', f'sub-All', f'sub-All_gradients_space-fsaverag5_confspec-{confspec}.npy')
    #   op.join(bids_folder_orig,'derivatives', 'gradients','reference_gradients_margulies16_space-fsaverage5_N-10.npy')) # Margulies gradient as reference ?!
    g_ref = np.load(g_ref_fn)
    g_ref_fil = g_ref[:,mask].T  # np.shape(g_ref) = (10,20484)

    n_components = 10
    gm = GradientMaps(n_components=n_components, alignment='procrustes' ) 
    gm.fit(cm_filtered ,reference=g_ref_fil)

    # save results
    np.save(op.join(target_dir,f'sub-{sub}_lambdas_space-fsaverag5_n10{specification}.npy'), gm.lambdas_) # save all together
    gm_= gm.gradients_.T 
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_gradients_space-fsaverag5_n10{specification}.npy'), grad) # save all together
    gm_ = gm.aligned_.T
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_g-aligned_space-fsaverag5_n10{specification}.npy'), grad) # save all together    
    print(f'finished sub-{sub} ses-{ses} task-{task} - cleaned - {confspec}: gradients saved')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--confspec', default='32P')
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk')
    #parser.add_argument('--specification', default='')

    cmd_args = parser.parse_args()
    main(cmd_args.subject, cmd_args.confspec, cmd_args.bids_folder)

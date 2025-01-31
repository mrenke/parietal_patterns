# all steps of the gradient generation process combined,
# hence: needs freesurfer directory for 1. fsavTofsav5 (laoding in), then fsav5Tofsnative (save)
# OQ: 
# - save the whole gradient object? (lambdas - explained variance, etc.)
# - save filtered masks ?

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
from utils import get_basic_mask, fit_correlation_matrix_unfiltered #saveGradToNPFile, npFileTofs5Gii,fsav5Tofsnative

bids_folder_ref = '/mnt_03/ds-dnumrisk'

def main(sub,task,ses,bids_folder, specification='', n_components=10):

    #sub = '%02d' % int(sub)
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses}')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    #CM unfiltered generated before?
    cm_file = op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_ses-{ses}_task-{task}_CM-unfiltered.npy')
    if (os.path.exists(cm_file) == False):
        fit_correlation_matrix_unfiltered(sub,ses,task,bids_folder = bids_folder)
    cm = np.load(cm_file)

    # filter out nodes that are not connected to the rest
    cc_mask_file = op.join(target_dir,f'sub-{sub}_ses-{ses}_task-{task}_cc-mask_space-fsaverag5.npy')
    if (os.path.exists(cc_mask_file) == False):
        cc = connected_components(cm)
        mask_cc = cc[1] == 0 # all nodes in 0 belong to the largest connected component, check #-components in cc[0]
        np.save(cc_mask_file, mask_cc) # save all together
        print('connected components derived & mask saved')    
    mask_cc = np.load(cc_mask_file)
    mask, labeling_noParcel = get_basic_mask()
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask
    cm_filtered = cm[mask_cc, :][:, mask_cc]
    print('connectivty matrix loaded and filtered with cc_mask')    

    # load in reference gradient and apply same filter
    g_ref = np.load(op.join(bids_folder_ref,'derivatives', 'gradients','sub-All', 'sub-All_gradients_N-10.npy')) # 
    print(np.shape(g_ref))
    g_ref_fil = g_ref[:,mask] # np.shape(g_ref) = (10,20484)

    # now perform embedding on cleaned data + alignment
    print(f'start fitting gradintes now')
    gm = GradientMaps(n_components=n_components,alignment='procrustes') # defaults: approacch = 'dm', kernel = None
    gm.fit(cm_filtered,reference=g_ref_fil.T)
    print(f'gradients generated')
    
    # save results
    np.save(op.join(target_dir,f'sub-{sub}_ses-{ses}_task-{task}_lambdas_space-fsaverag5_n10{specification}.npy'), gm.lambdas_) # save all together
    gm_= gm.gradients_.T 
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_ses-{ses}_task-{task}_gradients_space-fsaverag5_n10{specification}.npy'), grad) # save all together
    gm_ = gm.aligned_.T
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_ses-{ses}_task-{task}_g-aligned_space-fsaverag5_n10{specification}.npy'), grad) # save all together    
    print(f'finished sub-{sub} ses-{ses} task-{task}: gradients saved')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--task', default=None)
    parser.add_argument('--session', default=1, type=int)  
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--specification', default='')

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.task, cmd_args.session, cmd_args.bids_folder, 
         specification = cmd_args.specification)

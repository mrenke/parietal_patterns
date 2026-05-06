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
from my_utils import get_basic_mask, fit_correlation_matrix_unfiltered

def main(sub, ses, bids_folder_cm, bids_folder_ref, specification, kernel_name, z_transf, n_components=10):

    if kernel_name == 'None':
        kernel = None
    else:
        kernel = kernel_name

    sub = '%02d' % int(sub)
    target_dir = op.join(bids_folder_cm, 'derivatives', 'gradients.glmsingle', f'sub-{sub}')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Define the stimulus types
    stimulus_types = ['1', '2'] # '1', '2'

    for stim in stimulus_types:
        print(f'Processing stimulus: {stim}')       

        #CM unfiltered generated before?
        cm_file = op.join(bids_folder_cm, 'derivatives', 'correlation_matrices.glmsingle', f'sub-{sub}_ses-{ses}_stimulus-{stim}_betas_space-fsav5.npy')
        if (os.path.exists(cm_file) == False):
            #fit_correlation_matrix_unfiltered(sub, bids_folder = '/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk')
            print(f"Correlation matrix missing for sub-{sub}, stim {stim}. Skipping.")
            continue

        cm_notz = np.load(cm_file)

        if z_transf:
            # Apply Fisher z-transform (arctanh) to normalize correlations
            cm = np.arctanh(cm_notz) # leave it in arctanh space

            # Replace NaN and Inf values with 0
            cm[np.isnan(cm)] = 0
            cm[np.isinf(cm)] = 0
        
        else:
            cm = cm_notz
        
        specif = f'z_transf-{z_transf}'
        
        # filter out nodes that are not connected to the rest
        cc_mask_file = op.join(target_dir,f'sub-{sub}_cc-mask_space-fsaverag5_stim-{stim}_betas_kernel-{kernel_name}.npy')
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
        g_ref = np.load(op.join(bids_folder_ref, 'dataset-dnumrisk_sub-All_gradients_kernel-normalized_angle_ztransf-True_avMethod-tanH.npy')) #  'derivatives', 'marg_gradients', 'hemi_combined_marg01-10_fsav5_unmasked.npy'
        print(np.shape(g_ref))
        g_ref_fil = g_ref[:,mask] # np.shape(g_ref) = (10,20484)

        # now perform embedding on cleaned data + alignment
        print(f'start fitting gradintes now')
        gm = GradientMaps(n_components=n_components, alignment='procrustes', approach = 'dm', kernel = kernel, random_state=0) # defaults: approacch = 'dm', kernel = None
        gm.fit(cm_filtered,reference=g_ref_fil.T)
        print(f'finished sub-{sub}: gradients generated')
        
        # save results
        np.save(op.join(target_dir,f'sub-{sub}_lambdas_space-fsaverag5_n10{specification}_stimulus-{stim}_betas_kernel-{kernel_name}{specif}.npy'), gm.lambdas_) # save all together
        gm_= gm.gradients_.T 
        grad = [None] * n_components
        for i, g in enumerate(gm_): # gm.gradients_.T
            grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
        np.save(op.join(target_dir,f'sub-{sub}_gradients_space-fsaverag5_n10{specification}_stimulus-{stim}_betas_kernel-{kernel_name}{specif}.npy'), grad) # save all together
        gm_ = gm.aligned_.T
        grad = [None] * n_components
        for i, g in enumerate(gm_): # gm.gradients_.T
            grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
        np.save(op.join(target_dir,f'sub-{sub}_g-aligned_space-fsaverag5_n10{specification}_stimulus-{stim}_betas_kernel-{kernel_name}{specif}.npy'), grad) # save all together


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1, type=int)  
    parser.add_argument('--bids_folder_cm', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-numrisk')
    parser.add_argument('--bids_folder_ref', default='/mnt_AdaBD_largefiles/Data/DNumrisk_Data/connectivity_references') # /mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk
    parser.add_argument('--specification', default='')
    parser.add_argument('--z_transf', action='store_true')
    parser.add_argument('--kernel_name', default='normalized_angle')

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.session, cmd_args.bids_folder_cm, cmd_args.bids_folder_ref,
         cmd_args.specification, cmd_args.kernel_name, cmd_args.z_transf)
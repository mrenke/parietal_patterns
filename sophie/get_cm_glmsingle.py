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
from my_utils import cleanTS , get_basic_mask
from nipype.interfaces.freesurfer import SurfaceTransform # needs the fsaverage & fsaverage5 in ..derivatives/freesurfer folder!
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

# for plotting surface map
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from utils_old import get_events_confounds,surfTosurf

ses = 1

def main(sub, bids_folder_input, bids_folder_output, stim):
    sub = f'{int(sub):02d}'
    target_folder = op.join(bids_folder_output, 'derivatives', f'correlation_matrices.glmsingle')
    os.makedirs(target_folder, exist_ok=True)

    hemi_combined = [None] * 2
    for i, hemi in enumerate(['L', 'R']):
        filename = op.join(bids_folder_input, f'glm_stim{stim}.denoise', f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-1_task-magjudge_space-fsaverage5_stim-{stim}_hemi-{hemi}.func.gii') #{stim}     
        
        # Check if the file exists
        if not os.path.exists(filename):
            print(f"Beta file missing for sub-{sub}, hemi {hemi}. Skipping this subject.")
            return  # exit the main function early
        
        hemi_combined[i] = nib.load(filename).agg_data()
    print(f"Hemisphere {hemi} shape: {hemi_combined[i].shape}")
    hemi_combined = np.vstack(hemi_combined)

    # Build Destrieux parcellation and mask
    mask, labeling_noParcel = get_basic_mask()

    print(f'size of mask: {mask.shape}')

    print(f'Beta time series loaded for sub-{sub}')

    seed_ts = hemi_combined[mask]
    correlation_measure = ConnectivityMeasure(kind='correlation')
    
    # Compute the correlation matrix
    correlation_matrix = correlation_measure.fit_transform([seed_ts.T])[0]  # fit_transform returns a list of matrices

    # Apply Fisher z-transform (arctanh) to normalize correlations
    correlation_matrix_z = np.arctanh(correlation_matrix) # leave it in arctanh space

    # Replace NaN and Inf values with 0
    correlation_matrix_z[np.isnan(correlation_matrix_z)] = 0
    correlation_matrix_z[np.isinf(correlation_matrix_z)] = 0
    
    # Save the computed correlation matrix
    np.save(op.join(target_folder, f'sub-{sub}_ses-{ses}_stimulus-{stim}_betas_space-fsav5.npy'), correlation_matrix_z)
    print(f'Raw connectivity matrix estimated and saved for sub {sub} and stim {stim}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder_input', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-numrisk/derivatives/')
    parser.add_argument('--bids_folder_output', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-numrisk')
    parser.add_argument('--stim', default=1)

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder_input, cmd_args.bids_folder_output, cmd_args.stim
          )
    
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
from numrisk.fmri_analysis.gradients.utils_old import get_events_confounds,surfTosurf

ses = 1  # Session number
stim_duration = 3  # in seconds
TR = 2.3  # Repetition Time in seconds (sampling rate)
n_components = 3

def main(sub, bids_folder):
    target_folder = op.join(bids_folder, 'derivatives_02', 'correlation_matrices')
    os.makedirs(target_folder, exist_ok=True)
    #target_folder_gm = op.join(bids_folder, 'derivatives_02', 'gradients')

    # Build Destrieux parcellation and mask
    mask, labeling_noParcel = get_basic_mask()

    print(f'size of mask: {mask.shape}')

    # Get the cleaned time series for all types
    stimulus_1_combined, stimulus_2_combined, remaining_combined = cleanTS(sub, ses=ses, bids_folder=bids_folder, stim_duration=stim_duration, TR=TR)

    ts_types = {
        'stimulus_1': stimulus_1_combined,
        'stimulus_2': stimulus_2_combined,
        'remaining': remaining_combined
    }

    print(f'cleaned time series loaded for sub-{sub}')

    for ts_type, clean_ts in ts_types.items():
        seed_ts = clean_ts[mask]
        correlation_measure = ConnectivityMeasure(kind='correlation')
        np.save(op.join(target_folder_mask, f'sub-{sub}_ses-{ses}_unfiltered_{ts_type}_space-fsav5.npy'), correlation_measure)
        print(f'raw connectivity matrix estimated for {ts_type}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')



    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder, 
          )
#import cortex
from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
import math
from nilearn import signal
import pandas as pd
from nipype.interfaces.freesurfer import SurfaceTransform # needs the fsaverage & fsaverage5 in ..derivatives/freesurfer folder!
from nilearn import datasets
import matplotlib.colors as colors

# for plotting to the surface map
from brainspace.utils.parcellation import map_to_labels
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from utils_old import get_events_confounds,surfTosurf

def cleanTS(sub, ses =1, remove_task_effects = False, runs = range(1, 7),space = 'fsaverage5', bids_folder='/mnt_03/ds-dnumrisk', task = 'magjudge', stim_duration = 3, TR = 2.3):
    # load in data as timeseries and regress out confounds (for each run sepeprately)

    fmriprep_confounds_include = [
    'global_signal', 'dvars', 'framewise_displacement', 'trans_x',
    'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
    ] # 

    # get number of vertices
    if space == 'fsaverage5':
        number_of_vertex = 20484  # 'fsaverage5', 10242 * 2
    elif space == 'fsaverage':
        number_of_vertex = 327684  # 'fsaverage', 163842 * 2
    elif space == 'fsnative': # takes way to long to estimate CC
        timeseries = [None] * 2
        for i, hemi in enumerate(['L', 'R']): # have to load in both hemispheres to get the number of vertices (can be different for L&R)
            ex_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
            f'sub-{sub}_ses-{ses}_task-{task}run-1_space-{space}_hemi-L_bold.func.gii')
            timeseries[i] = nib.load(ex_file).agg_data()
        timeseries = np.vstack(timeseries)
        number_of_vertex = timeseries.shape[0]

    # Dictionary to store each cleaned run separately
    clean_ts_runs = {}

    for run in runs:
        try:
            timeseries = [None] * 2
            for i, hemi in enumerate(['L', 'R']):
                filename = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
                f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')        
                timeseries[i] = nib.load(filename).agg_data()
            timeseries = np.vstack(timeseries) # (20484, 135)

            fmriprep_confounds_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv')
            fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
            #fmriprep_confounds= fmriprep_confounds.fillna(method='bfill') # deprecated
            fmriprep_confounds= fmriprep_confounds.bfill()

            if remove_task_effects:
                dm = get_events_confounds(sub, ses, run, bids_folder)
                regressors_to_remove = pd.concat([dm.reset_index(drop=True), fmriprep_confounds], axis=1)
            else:
                regressors_to_remove = fmriprep_confounds
            clean_ts = signal.clean(timeseries.T, confounds=regressors_to_remove).T

            #clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1) # dont want to concatenate the TS just there

            # Store each cleaned run separately
            clean_ts_runs[run] = clean_ts

        except Exception as e:
            print(f"Error processing run {run} for sub-{sub}: {e} \nSkipping this run.")

    # After cleaning the time series for each run, now read the event files and dissect the time series based on stimuli

    stimulus_1_ts = []
    stimulus_2_ts = []
    remaining_ts = []

    for run in runs:
        try:
            # Load the event file for the current run
            event_file = os.path.join(bids_folder, f'sub-{sub}', f'ses-{ses}', 'func',
                                      f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv')
            df_events = pd.read_csv(event_file, sep='\t')

            # Exclude rows where trial number is 0
            df_events = df_events[df_events['trial_nr'] != 0]

            # Get onsets for stimulus 1 and stimulus 2
            stimulus_1_onsets = df_events[df_events['trial_type'] == 'stimulus 1']['onset'].values
            stimulus_2_onsets = df_events[df_events['trial_type'] == 'stimulus 2']['onset'].values

            # Get the duration (3s after each stimulus onset) and the corresponding indices for the time series
            frame_indices_1 = [(onset / TR) for onset in stimulus_1_onsets]  # Convert onset to frame indices
            frame_indices_2 = [(onset / TR) for onset in stimulus_2_onsets]

            # Extract the time series for stimulus 1 and stimulus 2 (3s after the onset)
            ts_stimulus_1 = []
            ts_stimulus_2 = []
            ts_remaining = []

            all_indices = set(range(clean_ts_runs[run].shape[1]))

            for onset_frame in frame_indices_1:
                start_index = math.ceil(onset_frame)
                end_index = math.floor((start_index + (stim_duration / TR)) + 1)
                ts_stimulus_1.append(clean_ts_runs[run][:, start_index:end_index])
                all_indices -= set(range(start_index, end_index))

            for onset_frame in frame_indices_2:
                start_index = math.ceil(onset_frame)  # Round onset frame up
                end_index = math.floor((start_index + (stim_duration / TR)) + 1)  # Round end frame down
                ts_stimulus_2.append(clean_ts_runs[run][:, start_index:end_index])
                all_indices -= set(range(start_index, end_index))

            # Extract remaining time series
            ts_remaining.append(clean_ts_runs[run][:, list(all_indices)])

            # Concatenate the time series for each stimulus across all runs
            stimulus_1_ts.append(np.concatenate(ts_stimulus_1, axis=1))  # Stimulus 1 time series
            stimulus_2_ts.append(np.concatenate(ts_stimulus_2, axis=1))  # Stimulus 2 time series
            remaining_ts.append(np.concatenate(ts_remaining, axis=1))  # Remaining time series

        except Exception as e:
            print(f"Error processing events for run {run}: {e}")

    # Print shapes for debugging
    print(f"Stimulus 1 combined shape: {[ts.shape for ts in stimulus_1_ts]}")
    print(f"Stimulus 2 combined shape: {[ts.shape for ts in stimulus_2_ts]}")
    print(f"Remaining combined shape: {[ts.shape for ts in remaining_ts]}")

    stimulus_1_combined = np.concatenate(stimulus_1_ts, axis=1)  # Concatenate across time (axis=1)
    stimulus_2_combined = np.concatenate(stimulus_2_ts, axis=1)  # Concatenate across time (axis=1)
    remaining_combined = np.concatenate(remaining_ts, axis=1)  # Concatenate across time (axis=1)

    return stimulus_1_combined, stimulus_2_combined, remaining_combined


def fit_correlation_matrix_unfiltered(sub, bids_folder, ts_type='stimulus_1'):
    mask, labeling_noParcel = get_basic_mask()
    stimulus_1_combined, stimulus_2_combined, remaining_combined = cleanTS(sub, bids_folder=bids_folder)
    
    if ts_type == 'stimulus_1':
        clean_ts = stimulus_1_combined
    elif ts_type == 'stimulus_2':
        clean_ts = stimulus_2_combined
    elif ts_type == 'remaining':
        clean_ts = remaining_combined
    else:
        raise ValueError("Invalid ts_type. Choose from 'stimulus_1', 'stimulus_2', or 'remaining'.")

    seed_ts = clean_ts[mask]

    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    # Apply Fisher z-transform (arctanh) to normalize correlations
    cm_z = np.arctanh(cm) # leave it in arctanh space
    # Replace NaN and Inf values with 0
    cm_z[np.isnan(cm_z)] = 0
    cm_z[np.isinf(cm_z)] = 0
    print(f'sub-{sub}: raw connectivity matrix estimated')    
    np.save(op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_unfiltered_{ts_type}.npy'), cm_z)


def get_basic_mask():
    atlas = datasets.fetch_atlas_surf_destrieux()
    regions = atlas['labels'].copy()
    masked_regions = [b'Medial_wall', b'Unknown']
    masked_labels = [regions.index(r) for r in masked_regions]
    for r in masked_regions:
        regions.remove(r)
    labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
    labeling_noParcel = np.arange(0,len(labeling),1,dtype = int)     # Map gradients to original parcels
    mask = ~np.isin(labeling, masked_labels)
    return mask, labeling_noParcel

def get_glasser_parcels(base_folder='/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/atlases_parcellations', space='fsaverage5'):
    atlas_left = nib.load(op.join(base_folder,f'lh_space-{space}.HCPMMP1.gii')).agg_data()
    atlas_right =  nib.load(op.join(base_folder,f'rh_space-{space}.HCPMMP1.gii')).agg_data()

    labeling = np.concatenate([(atlas_left+1000), (atlas_right+2000)]) # unique labels for left and right!
    mask = ~np.isin(labeling, [1000,2000]) # non-cortex region (unknow and medial wall) have label 0, hence 1000 & 2000 in my variation labels L/R
    # mask.sum() == len(labeling[(labeling != 1000) & (labeling != 2000)]) 
    return mask, labeling

def get_glasser_CAatlas_mapping(datadir = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/atlases_parcellations/ColeAnticevicNetPartition'):
    glasser_CAatlas_mapping = pd.read_csv(op.join(datadir,'cortex_parcel_network_assignments.txt'),header=None)
    glasser_CAatlas_mapping.index.name = 'glasser_parcel'
    glasser_CAatlas_mapping = glasser_CAatlas_mapping.rename({0:'ca_network'},axis=1)

    CAatlas_names = pd.read_csv(op.join(datadir,'network_label-names.csv'),index_col=0)
    CAatlas_names = CAatlas_names.set_index('Label Number')
    CAatlas_names = CAatlas_names.sort_index(level='Label Number')
    return glasser_CAatlas_mapping, CAatlas_names

def get_GMmargulies_cmap(skewed=True): 
    # proportion of the two colormaps, defines how much space is taken by each
    first = int((128*2)-np.round(255*(1.-0.90)))
    second = (256-first)
    first = first if skewed else second
    colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
    colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))

    # combine them and build a new colormap
    cols = np.vstack((colors2,colors3))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)
    return mymap
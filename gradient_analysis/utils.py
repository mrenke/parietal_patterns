from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from nipype.interfaces.freesurfer import SurfaceTransform # needs the fsaverage & fsaverage5 in ..derivatives/freesurfer folder!
from nilearn import datasets
import sys
import glob

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

def cleanTS(sub, ses =1, task ='magjudge',runs = range(1, 7),space = 'fsaverage5', bids_folder='/Users/mrenke/data/ds-dnumrisk'): #  'magjudge'
    # load in data as timeseries and regress out confounds (for each run sepeprately)
    if bids_folder.endswith('ds-smile1'):
        study = 'smile1'
        if task == 'magjudge':
            runs = range(1, 4)
        elif task =='rest':
            runs = [1]
    elif bids_folder.endswith('ds-numrisk'):
        study = 'miguel'

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                    'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
                                    ] # 
    number_of_vertices = 20484 if space == 'fsaverage5' else sys.exit("currently only space='fsaverage5'implemented ")
    clean_ts_runs = np.empty([number_of_vertices,0])
    for run in runs: # loop over runs and concatenate timeseries
        #try:
            timeseries = [None] * 2
            for i, hemi in enumerate(['L', 'R']):
                if study == 'smile1':
                    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 
                    filename =  op.join(fmriprep_folder, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')   #_ses-{ses}
                    timeseries[i] = nib.load(filename).agg_data()
                elif study == 'miguel':
                    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', 'func') # f'ses-{ses}', 
                    filename_pattern = op.join(fmriprep_folder, f"sub-{sub}_task-{task}_acq-*_run-{run}_space-{space}_hemi-{hemi}.func.gii")
                    timeseries[i] = nib.load(glob.glob(filename_pattern)[0]).agg_data()
            timeseries = np.vstack(timeseries) # (20484, 135)

            # confounds
            if study == 'smile1':
                fmriprep_confounds_file = op.join(fmriprep_folder,f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv') # _ses-{ses} timeseries
                fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
            elif study == 'miguel':
                fmriprep_confounds_filename_pattern = op.join(fmriprep_folder, f"sub-{sub}_task-{task}_acq-*_run-{run}_desc-confounds_regressors.tsv")
                fmriprep_confounds = pd.read_table(glob.glob(fmriprep_confounds_filename_pattern)[0])[fmriprep_confounds_include] 
            fmriprep_confounds= fmriprep_confounds.bfill()

            regressors_to_remove = fmriprep_confounds # remove_task_effects not implemented here (check dnumrisk)
            clean_ts = signal.clean(timeseries.T, confounds=regressors_to_remove).T
            clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
        #except:
            #print(f'sub-{sub}, run-{run} makes problems') # (prob. confounds ts not there){fmriprep_confounds_file} \n skipping that run') # for sub 5,47,53,62

    return clean_ts_runs

def fit_correlation_matrix_unfiltered(sub,ses,task,bids_folder):
    mask, labeling_noParcel = get_basic_mask()
    clean_ts = cleanTS(sub,ses,task,bids_folder=bids_folder) # checks if fsav5-file exists, if not, creates it
    seed_ts = clean_ts[mask]

    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    print(f'sub-{sub} ses-{ses} task-{task}: raw connectivity matrix estimated')    
    np.save(op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_ses-{ses}_task-{task}_CM-unfiltered.npy'), cm)

# ATLAS stuff

def get_glasser_parcels(base_folder='/mnt_03/diverse_neuralData/atlases_parcellations', space='fsaverage'):
    atlas_left = nib.load(op.join(base_folder,f'lh_space-{space}.HCPMMP1.gii')).agg_data()
    atlas_right =  nib.load(op.join(base_folder,f'rh_space-{space}.HCPMMP1.gii')).agg_data()

    labeling = np.concatenate([(atlas_left+1000), (atlas_right+2000)]) # unique labels for left and right!
    mask = ~np.isin(labeling, [1000,2000]) # non-cortex region (unknow and medial wall) have label 0, hence 1000 & 2000 in my variation labels L/R
    # mask.sum() == len(labeling[(labeling != 1000) & (labeling != 2000)]) 
    return mask, labeling

def get_glasser_CAatlas_mapping(datadir = '/mnt_03/diverse_neuralData/atlases_parcellations/ColeAnticevicNetPartition'):
    glasser_CAatlas_mapping = pd.read_csv(op.join(datadir,'cortex_parcel_network_assignments.txt'),header=None)
    glasser_CAatlas_mapping.index.name = 'glasser_parcel'
    glasser_CAatlas_mapping = glasser_CAatlas_mapping.rename({0:'ca_network'},axis=1)

    CAatlas_names = pd.read_csv(op.join(datadir,'network_label-names.csv'),index_col=0)
    CAatlas_names = CAatlas_names.set_index('Label Number')
    CAatlas_names = CAatlas_names.sort_index(level='Label Number')
    
    return glasser_CAatlas_mapping, CAatlas_names
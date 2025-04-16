import argparse
from nilearn.connectome import ConnectivityMeasure
import os.path as op
import os
from nilearn import signal
import numpy as np

import nibabel as nib
import pandas as pd
from nilearn import signal
from utils import get_basic_mask

confspec = '36P' # kate2
# you seem to have done the acompcor one but other common strategies don't include acompcor but white matter and CSF
mov_params = ['trans_x','trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
general_params = ['csf','white_matter','global_signal']
base_params = mov_params + general_params

fmriprep_confounds_include = base_params.copy()
for param in base_params: # add derivative1 and power2 to all realignment parameters and others 
    fmriprep_confounds_include.append(param + '_derivative1')
    fmriprep_confounds_include.append(param + '_power2')
    fmriprep_confounds_include.append(param + '_derivative1_power2')



def cleanTS(sub, ses =1, task ='magjudge',runs = range(1, 7),space = 'fsaverage5', bids_folder='/mnt_03/ds-dnumrisk', fmriprep_confounds_include = fmriprep_confounds_include): #  'magjudge'
    print(fmriprep_confounds_include)

    number_of_vertices = 20484
    clean_ts_runs = np.empty([number_of_vertices,0])
    for run in runs:
        timeseries = [None] * 2
        for i, hemi in enumerate(['L', 'R']):  
            fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 
            filename =  op.join(fmriprep_folder, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')   #_ses-{ses}
            timeseries[i] = nib.load(filename).agg_data()        
        timeseries = np.vstack(timeseries) # (20484, N_timepoints)
        # load in and remove confounds
        fmriprep_confounds_file = op.join(fmriprep_folder,f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv') # _ses-{ses} timeseries
        fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
        fmriprep_confounds= fmriprep_confounds.bfill()
        clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds).T
        clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
    return clean_ts_runs

cc_filter= False

def main(sub, bids_folder_in, bids_folder_out, ses=1, task='magjudge', confspec='36P'):  
    sub = f'{int(sub):02d}'
    specification = confspec

    clean_ts = cleanTS(sub, bids_folder=bids_folder_in)
    mask, labeling_noParcel = get_basic_mask()
    seed_ts = clean_ts[mask]

    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    fn = op.join(bids_folder_out, 'derivatives', 'correlation_matrices.tryNoHalo', f'sub-{sub}_ses-{ses}_task-magjudge_confspec-{confspec}_CM-unfiltered.npy')
    np.save(fn, cm)
    print(f'sub-{sub} ses-{ses} task-{task} conf-{confspec}: raw connectivity matrix estimated & saved to {fn}')    

    # filter out nodes that are not connected to the rest | the next steps... 
    if cc_filter:
        target_dir = op.join(bids_folder_out, 'derivatives', 'gradients.tryNoHalo', f'sub-{sub}', f'ses-1')
        if not op.exists(target_dir):
            os.makedirs(target_dir)

        #bids_folder_ref = bids_folder_in
        from scipy.sparse.csgraph import connected_components
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder_input', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--bids_folder_output', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk')

    cmd_args = parser.parse_args()
    main(cmd_args.subject, cmd_args.bids_folder_input, cmd_args.bids_folder_output)

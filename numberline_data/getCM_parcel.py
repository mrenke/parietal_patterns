import argparse
import os
from nilearn.connectome import ConnectivityMeasure
from brainspace.utils.parcellation import reduce_by_labels
import numpy as np
import os.path as op
from nilearn import signal
import nibabel as nib
import pandas as pd

def cleanTS(sub, fmriprep_confounds_include, bids_folder, ses =1, 
        task ='digitorder',space = 'fsaverage5',TR=2.1,
        scrub_thresh= 0.3, # scrubbing=True, run_FD_filter = True, frames_per_run_thresh=104, bp_filtering=True, 
        lower_bpf=0.01, upper_bpf=0.08): 

    print(fmriprep_confounds_include)
    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 

    timeseries = [None] * 2
    for i, hemi in enumerate(['L', 'R']):  
        filename =  op.join(fmriprep_folder, f'sub-{sub}_ses-{ses}_task-{task}_space-{space}_hemi-{hemi}_bold.func.gii')   #_ses-{ses}
        timeseries[i] = nib.load(filename).agg_data()        
    timeseries = np.vstack(timeseries) # (20484, N_timepoints)
    
    # load in and remove confounds
    fmriprep_confounds_file = op.join(fmriprep_folder,f'sub-{sub}_ses-{ses}_task-{task}_desc-confounds_timeseries.tsv') # _ses-{ses} timeseries
    fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
    fmriprep_confounds= fmriprep_confounds.bfill()

            #if scrubbing:
    print('performing scrubbing with threshold', scrub_thresh)
    sample_mask = (pd.read_table(fmriprep_confounds_file)['framewise_displacement'] < scrub_thresh).to_numpy()
             #if bp_filtering:
    print('performing bandpass filtering')
    clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, 
        sample_mask=sample_mask, t_r = TR, standardize='zscore_sample',
        low_pass=upper_bpf, high_pass=lower_bpf).T

    return clean_ts


def main(subject, bids_folder,  confspec='36P', ses=1, task ='digitorder',space = 'fsaverage5'):
    
    target_folder_cm = op.join(bids_folder, 'derivatives', 'correlation_matrices.parcel')
    os.makedirs(target_folder_cm) if not op.exists(target_folder_cm) else None
    #sub = f'{int(subject):02d}'

    # definde confounds to include
    mov_params = ['trans_x','trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    if confspec == '36P':
        general_params = ['csf','white_matter','global_signal']
    elif confspec == '32P': # no global signal !
        general_params = ['csf','white_matter'] 
    base_params = mov_params + general_params
    fmriprep_confounds_include = base_params.copy()
    for param in base_params: # add derivative1 and power2 to all realignment parameters and others 
        fmriprep_confounds_include.append(param + '_derivative1')
        fmriprep_confounds_include.append(param + '_power2')
        fmriprep_confounds_include.append(param + '_derivative1_power2')

    clean_ts = cleanTS(subject, fmriprep_confounds_include, bids_folder, ses=ses, task = task, space = space)
    from numrisk.fmri_analysis.gradients.utils import get_glasser_parcels
    mask, labeling = get_glasser_parcels(space = space)  
    seed_ts = reduce_by_labels(clean_ts[mask], labeling[mask], axis=1, red_op='mean',dtype=float)
    
    # CM
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0]

    target_fn = op.join(target_folder_cm,f'sub-{subject}_ses-{ses}_confspec-{confspec}.npy') # _glasserParcel-{space}
    np.save(target_fn,cm) # 
    print(f'sub-{subject} - ses-{ses}: connectivity matrix estimated - saved to {target_fn}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numberline')
    parser.add_argument('--confspec', default='36P')
    parser.add_argument('--ses', default=1)

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder, 
        confspec=cmd_args.confspec, ses=cmd_args.ses)
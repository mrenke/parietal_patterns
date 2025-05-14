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
from nilearn.connectome import ConnectivityMeasure

 # Current implementation:
 # 36P, scrubbing, frames_per_run_thresh, BP-filtering
runs_per_sub_thresh = 2


def cleanTS(sub, fmriprep_confounds_include, bids_folder, 
        TR=2.3, ses =1, task ='magjudge',runs = range(1, 7),space = 'fsaverage5',
        scrubbing=True, scrub_thresh= 0.3,
        run_FD_filter = True, frames_per_run_thresh=104,
        bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08): #  'magjudge'

    print(fmriprep_confounds_include)
    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 

    if space == 'fsaverage5':
        number_of_vertices = 20484 
    elif space == 'fsaverage':
        number_of_vertices = 327684

    clean_ts_runs = np.empty([number_of_vertices,0])
    N_valid_runs = 0
    for run in runs:
        try: 
            timeseries = [None] * 2
            for i, hemi in enumerate(['L', 'R']):  
                filename =  op.join(fmriprep_folder, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')   #_ses-{ses}
                timeseries[i] = nib.load(filename).agg_data()        
            timeseries = np.vstack(timeseries) # (20484, N_timepoints)
            # load in and remove confounds
            fmriprep_confounds_file = op.join(fmriprep_folder,f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv') # _ses-{ses} timeseries
            fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
            fmriprep_confounds= fmriprep_confounds.bfill()

            if scrubbing:
                print('performing scrubbing with threshold', scrub_thresh)
                sample_mask = (pd.read_table(fmriprep_confounds_file)['framewise_displacement'] < scrub_thresh).to_numpy()
                usable_frames = np.sum(sample_mask == True)
                if run_FD_filter:
                    if usable_frames > frames_per_run_thresh:
                        print(f'run {run} has {usable_frames} usable frames')
                        #clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, sample_mask=sample_mask, t_r = TR, standardize='zscore_sample').T
                        if bp_filtering:
                            print('performing bandpass filtering')
                            clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, 
                                sample_mask=sample_mask, t_r = TR, standardize='zscore_sample',
                                low_pass=upper_bpf, high_pass=lower_bpf).T
                        clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
                        N_valid_runs += 1
                    else:
                        print(f'run {run} has {usable_frames} usable frames, not usable')
                else:
                    clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, sample_mask=sample_mask, t_r = TR, standardize='zscore_sample').T
                    clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)        
            else:
                clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, t_r = TR, standardize='zscore_sample').T
                clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
        except Exception as e:
            print(f"Error processing run {run} for sub-{sub}: {e} \nSkipping this run.")
            #print(f'sub-{sub}, run-{run} makes problems') # (prob. confounds ts not there){fmriprep_confounds_file} \n skipping that run') # for sub 5,47,53,62
        
    if N_valid_runs < runs_per_sub_thresh:
        print(f'sub-{sub} has {N_valid_runs} valid runs, not usable')
        
    return clean_ts_runs, N_valid_runs

cc_filter= False

def main(sub, bids_folder_in, bids_folder_out,  confspec='36P', ses=1, task='magjudge',
        scrubbing=True, scrub_thresh=0.3, 
        run_FD_filter=True, frames_per_run_thresh=104,
        bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08):  

    sub = f'{int(sub):02d}'

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

    clean_ts, N_valid_runs = cleanTS(sub,fmriprep_confounds_include=fmriprep_confounds_include, bids_folder=bids_folder_in, 
                scrubbing=scrubbing,  scrub_thresh=scrub_thresh,
                run_FD_filter=run_FD_filter, frames_per_run_thresh=frames_per_run_thresh,
                bp_filtering=bp_filtering, lower_bpf=lower_bpf, upper_bpf=upper_bpf,
                ) #, runs = range(1, 7)

    confspec += f'scrub{str(scrub_thresh)[2]}'  if scrubbing else confspec
    confspec += 'BPfilter' if bp_filtering else confspec
    confspec += f'runFD{str(frames_per_run_thresh)}' if run_FD_filter else confspec
    confspec += f'-{N_valid_runs}runs' if run_FD_filter else confspec

    mask, labeling_noParcel = get_basic_mask()
    seed_ts = clean_ts[mask]

    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    fn = op.join(bids_folder_out, 'derivatives', 'correlation_matrices.tryNoHalo', f'sub-{sub}_ses-{ses}_task-magjudge_confspec-{confspec}_CM-unfiltered.npy')
    np.save(fn, cm)
    print(f'sub-{sub} ses-{ses} task-{task} conf-{confspec}: raw connectivity matrix estimated & saved to {fn}')    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder_input', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--bids_folder_output', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk')
    parser.add_argument('--confspec', default='36P')
    cmd_args = parser.parse_args()
    main(cmd_args.subject, cmd_args.bids_folder_input, cmd_args.bids_folder_output, 
        confspec=cmd_args.confspec)

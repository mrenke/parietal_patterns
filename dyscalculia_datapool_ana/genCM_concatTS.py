# Generate Correlation matrices for over sessions&tasks concatenated time series 

# Problems: for numberline, not all subs have all sessions!
#  
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
import json

# Save usable frames info per subject, session & task combination
def append_usableFrames_to_csv(subject, ses_task_combi, N_usable_frames, csv_fn):
    if os.path.exists(csv_fn):
        df = pd.read_csv(csv_fn, index_col=['subject', 'ses_task_combi'])
    else:
        df = pd.DataFrame(columns=['subject', 'ses_task_combi', 'usable_frames']).set_index(['subject', 'ses_task_combi'])
    df.loc[(subject, ses_task_combi), 'usable_frames'] = N_usable_frames
    df.to_csv(csv_fn)

def get_cleanTS(sub,  bids_folder, 
            sessions, 
            task_names,
            #TR=2.3,
            space = 'fsaverage5', scrub_thresh= 0.3,
            lower_bpf=0.01, upper_bpf=0.08): #  

    # define counfounds to regress out:
    base_params =['trans_x','trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf','white_matter','global_signal']
    fmriprep_confounds_include = base_params.copy()
    for param in base_params: # add derivative1 and power2 to all realignment parameters and others 
        fmriprep_confounds_include.append(param + '_derivative1')
        fmriprep_confounds_include.append(param + '_power2')
        fmriprep_confounds_include.append(param + '_derivative1_power2')
    print(fmriprep_confounds_include)

    taskList = task_names.split('-') if '-' in task_names else [task_names]    
    sesList = sessions.split('-') if '-' in sessions else [int(sessions)]
    number_of_vertices = 20484  if space == 'fsaverage5' else NotImplementedError("only fsaverage5 space implemented!")
    cleanTS_all = np.empty([number_of_vertices,0])
    usable_frames_all = 0

    for ses in sesList:
        fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 
        for task in taskList:
            # get TR for task
            task_js = json.load(open(op.join(bids_folder,f'task-{task}_bold.json')))
            TR = task_js['RepetitionTime']
            runs = range(1,3) if task == 'magjudge' else [1]  # magjudge has multiple runs, other tasks only 1 run
            for run in runs: # only needed for magjudge from SMILE 
                run_spec = f'_run-{run}' if bids_folder.endswith('smile') else ''
                try: 
                    timeseries = [None] * 2
                    for i, hemi in enumerate(['L', 'R']):  
                        filename =  op.join(fmriprep_folder, f'sub-{sub}_ses-{ses}_task-{task}{run_spec}_space-{space}_hemi-{hemi}_bold.func.gii')   #_run-{run}
                        timeseries[i] = nib.load(filename).agg_data()        
                    timeseries = np.vstack(timeseries) # (20484, N_timepoints)
                    
                    # load in and remove confounds
                    fmriprep_confounds_file = op.join(fmriprep_folder,f'sub-{sub}_ses-{ses}_task-{task}{run_spec}_desc-confounds_timeseries.tsv') # _ses-{ses} timeseries
                    fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
                    fmriprep_confounds= fmriprep_confounds.bfill()

                    #scrubbing:
                    sample_mask = (pd.read_table(fmriprep_confounds_file)['framewise_displacement'] < scrub_thresh).to_numpy()
                    usable_frames = np.sum(sample_mask == True)

                    #bp_filtering:
                    clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, 
                                sample_mask=sample_mask, t_r = TR, standardize='zscore_sample',
                                low_pass=upper_bpf, high_pass=lower_bpf).T
                    cleanTS_all = np.append(cleanTS_all, clean_ts, axis=1)
                    usable_frames_all += usable_frames
                    print(f'sub-{sub} ses-{ses} task-{task} run-{run_spec}: {usable_frames} usable frames added.')
                except Exception as e:
                    print(f"Error processing ses {ses} & task {task} for sub-{sub}: {e} \nSkipping this run.")

    return cleanTS_all, usable_frames_all
    
def main(sub, bids_folder,  sessions, tasks='magjudge-placevalue-rest'):

    sub = f'{int(sub):02d}'

    target_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}')
    os.makedirs(target_folder, exist_ok=True)
    
    cleanTS_all, usable_frames_all = get_cleanTS(sub,  bids_folder, 
            sessions = sessions, task_names =tasks)
    append_usableFrames_to_csv(sub, f'ses-{sessions}_task-{tasks}', usable_frames_all, 
                  op.join(bids_folder, 'derivatives', 'correlation_matrices', 'usable_frames_per_subject.csv'))

    mask, labeling_noParcel = get_basic_mask()
    seed_ts = cleanTS_all[mask]
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    fn = op.join(target_folder, f'sub-{sub}_ses-{sessions}_task-{tasks}_funcCM.npy')
    np.save(fn, cm)
    print(f'sub-{sub} ses-{sessions} task-{tasks}: raw connectivity matrix estimated & saved to {fn}')    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-smile')
    parser.add_argument('--sessions', default='1')
    parser.add_argument('--tasks', default='magjudge')

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder, 
        sessions=cmd_args.sessions, 
        tasks=cmd_args.tasks,)  
import argparse
from nilearn.connectome import ConnectivityMeasure
import os.path as op
import os
from nilearn import signal
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import signal
from my_utils import get_basic_mask
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

 # Current implementation:
 # 36P, scrubbing, frames_per_run_thresh, BP-filtering
runs_per_sub_thresh = 2

def cleanTS(sub, fmriprep_confounds_include, bids_folder,
        TR=2.3, ses =1, task ='magjudge',runs = range(1, 7),space = 'fsaverage5',
        scrubbing=True, scrub_thresh= 0.3,
        run_FD_filter = True, frames_per_run_thresh=104,
        bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08): #  

    print(fmriprep_confounds_include)
    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 

    if space == 'fsaverage5':
        number_of_vertices = 20484 
    elif space == 'fsaverage':
        number_of_vertices = 327684

    clean_ts_runs = np.empty([number_of_vertices,0])
    N_valid_runs = 0
    total_usable_frames = 0

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

                print(f'run {run} has {usable_frames} usable frames')
                #clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, sample_mask=sample_mask, t_r = TR, standardize='zscore_sample').T
                if bp_filtering:
                    print('performing bandpass filtering')
                    clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds, 
                        sample_mask=sample_mask, t_r = TR, standardize='zscore_sample',
                        low_pass=upper_bpf, high_pass=lower_bpf).T
                clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
                N_valid_runs += 1
 
                total_usable_frames += usable_frames  
                
        except Exception as e:
            print(f"Error processing run {run} for sub-{sub}: {e} \nSkipping this run.")
            #print(f'sub-{sub}, run-{run} makes problems') # (prob. confounds ts not there){fmriprep_confounds_file} \n skipping that run') # for sub 5,47,53,62
        
    #if N_valid_runs < runs_per_sub_thresh:
    #    print(f'sub-{sub} has {N_valid_runs} valid runs, not usable')
        
    return clean_ts_runs, N_valid_runs, total_usable_frames

cc_filter= False

def grad_dispersion(bids_folder, kernel, sub, cm):

    cc_mask_file = op.join(bids_folder, 'derivatives', 'gradients.36Pscrub3BPfilterrunFD104', f'sub-{sub}', f'sub-{sub}_cc-mask_space-fsaverag5.npy')
    mask_cc = np.load(cc_mask_file)
    mask, labeling_noParcel = get_basic_mask()
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask
    cm_filtered = cm[mask_cc, :][:, mask_cc]
    print('connectivty matrix loaded and filtered with cc_mask')  

    n_components = 10
    mask, labeling_noParcel = get_basic_mask()

    bids_folder_reference = '/mnt_AdaBD_largefiles/Data/DNumrisk_Data/connectivity_references'

    # load in reference gradient and apply same filter
    g_ref = np.load(op.join(bids_folder_reference,'dataset-dnumrisk_sub-All_gradients_kernel-normalized_angle_ztransf-True_avMethod-tanH.npy')) # same labeling_noParcel as cm_unfiltered # 'derivatives','gradients.tryParams.36P',f'sub-{sub}', f'sub-{sub}_gradients_kernel-{kernel}_ztransf-True_avMethod-tanH.npy'
    g_ref = g_ref[:, mask] #

    # now perform embedding on cleaned data + alignment
    print(f'start fitting gradintes now')
    gm = GradientMaps(n_components=n_components, alignment='procrustes', approach='dm', kernel=kernel, random_state=0) # defaults: approach = 'dm', kernel = normalized_angle
    gm.fit(cm_filtered, reference=g_ref.T)
    gm_ = gm.aligned_.T
    grad = [None] * n_components
    diff = [None] * n_components
    sd = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
        diff[i] = np.nanmax(grad[i]) - np.nanmin(grad[i])
        sd[i] = np.nanstd(grad[i], ddof=1)
    
    return grad, diff, sd
    

def main(sub, bids_folder_input, bids_folder_output, kernel, steps, grad_nr, confspec='36P', ses=1, task='magjudge',
        scrubbing=True, scrub_thresh=0.3, 
        run_FD_filter=True, frames_per_run_thresh=104,
        bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08):  

    sub = f'{int(sub):02d}'

    mov_params = ['trans_x','trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    if confspec == '36P':
        general_params = ['csf','white_matter','global_signal']
    #elif confspec == '32P': # no global signal !
        #general_params = ['csf','white_matter'] 
    base_params = mov_params + general_params

    fmriprep_confounds_include = base_params.copy()
    for param in base_params: # add derivative1 and power2 to all realignment parameters and others 
        fmriprep_confounds_include.append(param + '_derivative1')
        fmriprep_confounds_include.append(param + '_power2')
        fmriprep_confounds_include.append(param + '_derivative1_power2')




    clean_ts, N_valid_runs, total_frames = cleanTS(sub,fmriprep_confounds_include=fmriprep_confounds_include, bids_folder=bids_folder_input, task=task, ses=ses, 
                scrubbing=scrubbing,  scrub_thresh=scrub_thresh,
                run_FD_filter=run_FD_filter, frames_per_run_thresh=frames_per_run_thresh,
                bp_filtering=bp_filtering, lower_bpf=lower_bpf, upper_bpf=upper_bpf
                ) #, runs = range(1, 7)

    confspec += f'scrub{str(scrub_thresh)[2]}'  if scrubbing else confspec
    confspec += 'BPfilter' if bp_filtering else confspec
    confspec += f'runFD{str(frames_per_run_thresh)}' if run_FD_filter else confspec
    confspec += f'-{N_valid_runs}runs' if run_FD_filter else confspec

    dic = {'usable_frames': [], 
           'diff': []}

    # Step 2: iteratively truncate concatenated time series
    n_frames_list = list(range(total_frames, 0, -steps))
    mask, labeling_noParcel = get_basic_mask()
        
    for n_frames in n_frames_list:
        truncated_ts = clean_ts[:, :n_frames]
        seed_ts = truncated_ts[mask]

        correlation_measure = ConnectivityMeasure(kind='correlation')
        c_m = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
        print(f'sub-{sub} ses-{ses} task-{task} conf-{confspec}: raw connectivity matrix estimated')    

        try:
            grad, diff, sd = grad_dispersion(bids_folder_output, kernel, sub, cm=c_m)


            dic['usable_frames'].append(n_frames)
            dic['diff'].append(diff[grad_nr-1])

            print(f'done with deducting {n_frames} frames')
        except ValueError as e:
            print(f"Skipping, {truncated_ts.shape[1]} frames remaining: gradient fit failed -> {e}")
            continue

    df = pd.DataFrame(dic)
    df.to_csv(op.join(bids_folder_output, 'derivatives', 'corr_usable-frames_grad-range_cm-sd', f'sub-{sub}.csv'))

    # x = df['usable_frames']
    # y = df['diff']
    # x_name = 'usable_frames'
    # y_name = 'diff'

    # slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # r, p = pearsonr(x, y)
    # print(r, p)

    # # Scatter plot
    # plt.figure(figsize=(6, 5))
    # plt.scatter(x, y, color='blue', alpha=0.7, label='Subjects per task')

    # # Fit and plot a regression line
    # slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # plt.plot(x, slope*x + intercept, color='red', label=f'Fit line: r={r_value:.2f}')

    # # Labels and title
    # plt.xlabel(f'{x_name}')
    # plt.ylabel(f'{y_name}')
    # plt.title(f"Scatter plot of {x_name} vs. {y_name}, p={p_value}")
    # plt.legend()
    # plt.tight_layout()
    # plot_dir = op.join(bids_folder_output, 'plots_and_ims')
    # os.makedirs(plot_dir, exist_ok=True)
    # plt.savefig(op.join(plot_dir, f'sub-{sub}_grad_vs_frames.png'), dpi=300)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default='60') # based on least framewise displacement
    parser.add_argument('--bids_folder_input', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--bids_folder_output', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk')    
    parser.add_argument('--steps', default=50, type=int)
    parser.add_argument('--grad_nr', default=1, type=int)
    parser.add_argument('--confspec', default='36P') # instead of 32P
    parser.add_argument('--task', default='magjudge')
    parser.add_argument('--ses', default=1, type=int)
    parser.add_argument('--kernel', default='normalized_angle')

    cmd_args = parser.parse_args()
    main(cmd_args.subject, cmd_args.bids_folder_input, cmd_args.bids_folder_output,
        cmd_args.kernel, 
        cmd_args.steps, cmd_args.grad_nr,
        confspec=cmd_args.confspec,
        task=cmd_args.task,
        ses=cmd_args.ses)
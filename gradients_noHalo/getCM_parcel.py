import argparse
from nilearn.connectome import ConnectivityMeasure
from brainspace.utils.parcellation import reduce_by_labels
import numpy as np
import os.path as op
from utils import get_glasser_parcels #local folder
from getCM_specConf import cleanTS #local folder
ses=1
space = 'fsaverage'

def main(subject, bids_folder_in, bids_folder_out,  confspec='32P', ses=1, task='magjudge',
        scrubbing=True, scrub_thresh=0.3, 
        run_FD_filter=True, frames_per_run_thresh=104,
        bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08): 
    
    target_folder_cm = op.join(bids_folder_out, 'derivatives', 'correlation_matrices.parcel')
    sub = f'{int(subject):02d}'
    
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

    # get cleanTS and average them per parcel
    mask, labeling = get_glasser_parcels()    # Get Glasser parcellation and mask
    clean_ts, N_valid_runs = cleanTS(sub,fmriprep_confounds_include=fmriprep_confounds_include, bids_folder=bids_folder_in, 
                space=space, # needed for average them per parcel
                scrubbing=scrubbing,  scrub_thresh=scrub_thresh,
                run_FD_filter=run_FD_filter, frames_per_run_thresh=frames_per_run_thresh,
                bp_filtering=bp_filtering, lower_bpf=lower_bpf, upper_bpf=upper_bpf,
                )    
    seed_ts = reduce_by_labels(clean_ts[mask], labeling[mask], axis=1, red_op='mean',dtype=float)
    
    # CM
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0]

    confspec += f'scrub{str(scrub_thresh)[2]}'  if scrubbing else confspec
    confspec += 'BPfilter' if bp_filtering else confspec
    target_fn = op.join(target_folder_cm,f'sub-{sub}_glasserParcel-{space}_confspec-{confspec}.npy')
    np.save(target_fn,cm) # 
    print(f'sub-{sub}: connectivity matrix estimated - saved to {target_fn}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder_input', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--bids_folder_output', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-dnumrisk')
    parser.add_argument('--confspec', default='32P')
    cmd_args = parser.parse_args()
    main(cmd_args.subject, cmd_args.bids_folder_input, cmd_args.bids_folder_output, 
        confspec=cmd_args.confspec)
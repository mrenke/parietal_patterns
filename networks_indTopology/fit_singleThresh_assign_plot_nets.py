import nilearn
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
from brainspace.utils.parcellation import map_to_labels
from infomap import Infomap

from numrisk.fmri_analysis.gradients.utils import get_basic_mask
mask, labeling_noParcel = get_basic_mask()
hemi = 'both'  

from fit_assign_consens_plot_nets import threshold_matrix, spatial_filtering,  assign_subject_communities_to_reference

def main(sub,bids_folder = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk',
        thresh_conn = 0.1 ,
        preferred_number_of_modules=None,
        confspec = '36Pscrub3BPfilter', hemi_to_plot='R',
        ses=1, task='magjudge'):  
    

    sub = f'{int(sub):02d}'
    source_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices.tryNoHalo') # .parcel
    target_folder = op.join(bids_folder,'derivatives','networks_infomap_singleThresh')
    plot_folder = op.join(bids_folder,'plots_and_ims','networks_infomap_singleThresh')

    cm_file = op.join(source_folder,f'sub-{sub}_ses-{ses}_task-{task}_confspec-{confspec}runFD104-6runs_CM-unfiltered.npy')
    cm_f = np.load(cm_file)

    cm_filtered = spatial_filtering(cm_f, bids_folder=bids_folder)

    conn_threshold = thresh_conn
    cm_thresh = threshold_matrix(cm_filtered, proportion=conn_threshold)
    N = cm_thresh.shape[0]
    im = Infomap(preferred_number_of_modules=None) # add flags like '--two-level' if needed
    for i in range(N):
        for j in range(i+1, N):
            w = cm_thresh[i, j]
            if w > 0:
                im.add_link(i, j, w)
    im.run()

    returned_nodes = np.array([node.node_id for node in im.nodes])
    returned_modules = np.array([node.module_id for node in im.nodes])
    full_module_mapping = np.full((N,), -1, dtype=int)  # -1 means unassigned
    full_module_mapping[returned_nodes] = returned_modules

    # asssign subject communities to reference
    fn_target_labels_caNets = op.join(target_folder, f'sub-average_target_labels_caNets_hemi-both_thresh-0.1_prefNmod-15_confspec-36Pscrub3BPfilter.npy')
    target_labels_caNets = np.load(fn_target_labels_caNets)
    relabeled_subject, assignments = assign_subject_communities_to_reference(full_module_mapping, target_labels_caNets,  jaccard_threshold=0.1)

    fn_mapping = op.join(target_folder, f'sub-{sub}_assignedModuleMapping_thresh-{thresh_conn}_prefNmod-{preferred_number_of_modules}_confspec-{confspec}.npy')
    np.save(fn_mapping, np.array(relabeled_subject))

    modules_fsav5 = np.full(mask.shape[0], np.nan, dtype=float)
    modules_fsav5[mask] = relabeled_subject

    from utils import plot_nets_CAcolors
    figure = plot_nets_CAcolors(modules_fsav5, hemi_to_plot='R')
    figure.suptitle(f'sub {sub}', y=0.75)
    plot_fn = op.join(plot_folder, f'sub-{sub}_thresh-{conn_threshold}_networks_infomap_hemi-R_confspec-{confspec}.png')
    figure.savefig(plot_fn, dpi=300, bbox_inches='tight')
    plt.close(figure)
    print(f'sub-{sub} thresh - {thresh_conn} finished!')

if __name__ == '__main__':  
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk', type=str, help='BIDS folder path')
    parser.add_argument('--confspec', default='36Pscrub3BPfilter', type=str, help='Configuration specification')
    parser.add_argument('--thresh_conn', default=0.1, type=float, help='Threshold for connection matrix')
    parser.add_argument('--preferred_number_of_modules', default=None, type=int, help='Preferred number of modules for Infomap')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, confspec=args.confspec, thresh_conn=args.thresh_conn, preferred_number_of_modules=args.preferred_number_of_modules)

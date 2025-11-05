import nilearn
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
from infomap import Infomap
import time

from numrisk.fmri_analysis.gradients.utils import get_basic_mask
mask, labeling_noParcel = get_basic_mask()
hemi = 'both'

conn_thresholds = [0.03, 0.04, 0.05, 0.1, 0.2, 0.4] 
conn_thresholds_string = "-".join([str(t) for t in conn_thresholds])
save_ind_thresh_labels = True
save_plot = True

from fit_assign_consens_plot_nets import threshold_matrix, spatial_filtering, assign_subject_communities_to_reference, get_consensus_assignment

def main(sub,bids_folder = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk',
        confspec = '36Pscrub3BPfilter', hemi_to_plot='R',
        ses=1, task='magjudge'):  
    
    start_time = time.time()
    print(f' start time: {time.ctime()}')

    sub = f'{int(sub):02d}'
    source_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices.tryNoHalo') # .parcel
    target_folder = op.join(bids_folder,'derivatives','networks_infomap_EG17nets')
    plot_folder = op.join(bids_folder,'plots_and_ims','networks_infomap_EG17nets')
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)


    #cm_file = op.join(source_folder,f'sub-{sub}_ses-{ses}_task-{task}_confspec-{confspec}runFD104-6runs_CM-unfiltered.npy')
    import glob
    cm_fn_pattern = op.join(source_folder, f"sub-{sub}_ses-{ses}_task-{task}_confspec-{confspec}runFD104-*runs_CM-unfiltered.npy")
    matching_files = glob.glob(cm_fn_pattern)
    cm_f = np.load(matching_files[0])

    cm_filtered = spatial_filtering(cm_f, bids_folder=bids_folder)

    fn_target_labels_EG17 = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/netAtlas_Gordon_17/netLabels_atlas-Gordon_17_fsaverage5_bothHemis.npy'
    target_labels_EG17 = np.load(fn_target_labels_EG17)
    target_labels_EG17 = target_labels_EG17[mask]

    sub_module_mappings_relabelled = []
    for conn_threshold in conn_thresholds:
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

        relabeled_subject, assignments = assign_subject_communities_to_reference(full_module_mapping, target_labels_EG17,  jaccard_threshold=0.1)
        sub_module_mappings_relabelled.append(relabeled_subject)
        elapsed_time = time.time() - start_time

        print(f'Subject {sub}, threshold {conn_threshold} Infomap run completed in {elapsed_time/60:.2f} minutes')

    if save_ind_thresh_labels:
        fn_mappings = op.join(target_folder, f'sub-{sub}_ses-{ses}_task-{task}_allThresholds_threshs-{conn_thresholds_string}_confspec-{confspec}.npy')
        np.save(fn_mappings, np.array(sub_module_mappings_relabelled))

    # can happen that small thresholds have very few modules, so we need to check if we have enough
    sub_module_mappings_relabelled_sufficient = []
    for thresh, mapping in zip(conn_thresholds, sub_module_mappings_relabelled):
        if len(np.unique(mapping)) > 5 or thresh > 0.09:  # More than five unique modules for smal thresholds
            sub_module_mappings_relabelled_sufficient.append(mapping)
        else:
            print(f'Skipping mapping with from threshold {thresh}')

    consensus_labels = get_consensus_assignment(sub_module_mappings_relabelled_sufficient)
    modules_fsav5 = np.full(mask.shape[0], np.nan, dtype=float)
    modules_fsav5[mask] = consensus_labels

    fn_consens_mapping = op.join(target_folder, f'sub-{sub}_ses-{ses}_task-{task}_consensusMapping_threshs-{conn_thresholds_string}_confspec-{confspec}.npy')
    np.save(fn_consens_mapping, np.array(consensus_labels))
    print(f'Consensus labels saved to {fn_consens_mapping}')

    if save_plot:
        from utils import get_Gordon17_cmap
        hemi_to_plot='R'
        i_hemi_to_plot = 0 if hemi_to_plot == 'L' else 1  

        from  nilearn.datasets import fetch_surf_fsaverage
        import nilearn.plotting as nplt 

        fsaverage = fetch_surf_fsaverage('fsaverage5') 
        views = ['medial','lateral','dorsal','posterior']
        surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
        bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left
        cmap = get_Gordon17_cmap()
        
        modules_fsav5_hemi = np.split(modules_fsav5,2)[i_hemi_to_plot]
        figure, axes = plt.subplots(nrows=1, ncols=len(views),figsize = (15,8), subplot_kw=dict(projection='3d'))
        for i,view in enumerate(views):
            colbar = True if view == 'posterior' else False
            nplt.plot_surf(surf_mesh=surf_mesh , surf_map= modules_fsav5_hemi, avg_method = 'median',# infl_right # pial_right
                    view= view, colorbar=colbar, #title=f'sub-{sub}, grad {n_grad+1}',
                    cmap=cmap,vmin = 0, vmax=17,
                    bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[i]) 
        figure.subplots_adjust(wspace=0.01)
        figure.suptitle(f'sub {sub}', y=0.75)

        plot_fn = op.join(plot_folder, f'sub-{sub}_ses-{ses}_task-{task}_networks_infomap_hemi-R_confspec-{confspec}.png')
        figure.savefig(plot_fn, dpi=300, bbox_inches='tight')
        plt.close(figure)
        
    elapsed_time = time.time() - start_time
    print(f'sub-{sub} finished in {elapsed_time/60:.2f} minutes!')
    print(f'end time: {time.ctime()}')

if __name__ == '__main__':  
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk', type=str, help='BIDS folder path')
    parser.add_argument('--confspec', default='36Pscrub3BPfilter', type=str, help='Configuration specification')
    parser.add_argument('--hemi_to_plot', default='R', type=str, help='Hemisphere to plot')
    parser.add_argument('--ses', default=1, type=int, help='Session number')
    parser.add_argument('--task', default='magjudge', type=str, help='Task name')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, confspec=args.confspec, hemi_to_plot=args.hemi_to_plot, ses=args.ses, task=args.task)

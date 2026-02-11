import nilearn
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
from infomap import Infomap

from numrisk.fmri_analysis.gradients.utils import get_basic_mask
mask, labeling_noParcel = get_basic_mask()
hemi = 'both'  # 
#conn_thresholds = ['0.02','0.05','0.1', '0.2', '0.3', '0.5']
#conn_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]  # Convert to float for thresholding
conn_thresholds = [0.03, 0.04, 0.05, 0.1, 0.2, 0.4] 
conn_thresholds_string = "-".join([str(t) for t in conn_thresholds])

import matplotlib.patches as mpatches
import hcp_utils as hcp
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt 

def plot_nets_CAcolors(modules_fsav5,  hemi_to_plot = 'R'):
    rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])

    grey = np.array([[0.5, 0.5, 0.5, 1.0]])  # RGBA format: grey with full opacity
    cmap_ca = ListedColormap( np.vstack([grey, rgb]))

    fsaverage = fetch_surf_fsaverage('fsaverage5') 
    views = ['medial','lateral','dorsal','posterior']
    cmap = cmap_ca #'Paired'#''viridis' # 

    i_hemi_to_plot = 0 if hemi_to_plot == 'L' else 1  
    modules_fsav5_hemi = np.split(modules_fsav5,2)[i_hemi_to_plot]

    surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
    bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left

    figure, axes = plt.subplots(nrows=1, ncols=len(views),figsize = (15,8), subplot_kw=dict(projection='3d'))
    for i,view in enumerate(views):
        colbar = True if view == 'posterior' else False
        nplt.plot_surf(surf_mesh=surf_mesh , surf_map= modules_fsav5_hemi, avg_method = 'median',# infl_right # pial_right
                view= view,cmap=cmap, colorbar=colbar, #title=f'sub-{sub}, grad {n_grad+1}',
                vmin = 0, vmax=12,
                bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[i]) 
    figure.subplots_adjust(wspace=0.01)

    return figure

def get_target_labels_caNets():
    from numrisk.fmri_analysis.gradients.utils import get_glasser_CAatlas_mapping, get_glasser_parcels
    mask_glasser, labeling_glasser = get_glasser_parcels(space = 'fsaverage5' )
    glasser_CAatlas_mapping, CAatlas_names = get_glasser_CAatlas_mapping()

    from brainspace.utils.parcellation import map_to_labels
    caNets_fsav5_mapping = map_to_labels(glasser_CAatlas_mapping['ca_network'].values , 
                                     labeling_glasser, mask=mask_glasser, fill=0) #, fill=np.nan) #grad_sub[n_grad-1]

    print(caNets_fsav5_mapping[mask].shape)
    return caNets_fsav5_mapping[mask]

def threshold_matrix(mat, proportion=0.1):
    n = mat.shape[0]
    mat = mat.copy()
    np.fill_diagonal(mat, 0)  # Remove diagonal

    thresh = np.percentile(mat[mat > 0], 100 - 100 * proportion)     # Find threshold value
    mat[mat < thresh] = 0
    return mat

def spatial_filtering(cm_f, bids_folder=  '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk'):
        
        source_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices.tryNoHalo') # .parcel

        from nilearn import datasets
        atlas = datasets.fetch_atlas_surf_destrieux()
        masked_labels = [atlas['labels'].index(r) for r in [b'Medial_wall', b'Unknown']]

        mask_hemi_L = ~np.isin(atlas['map_left'], masked_labels)
        mask_hemi_R = ~np.isin(atlas['map_right'], masked_labels)

        geo_dist_L = np.load(op.join(source_folder, f'geo_dist_fsav5_hemi-lh.npy'))
        geo_dist_L = geo_dist_L[np.ix_(mask_hemi_L, mask_hemi_L)]

        geo_dist_R = np.load(op.join(source_folder, f'geo_dist_fsav5_hemi-rh.npy'))
        geo_dist_R = geo_dist_R[np.ix_(mask_hemi_R, mask_hemi_R)]

        N_nodes_hemi_L  = mask_hemi_L.sum()
        cm_hemi_L = cm_f[:N_nodes_hemi_L, :N_nodes_hemi_L]  
        cm_hemi_R = cm_f[N_nodes_hemi_L:, N_nodes_hemi_L:]

        # spatial exclusion mask - eplace within hemisphere matrices with the spatially filtered ones
        distance_threshold = 30.0  # in mm
        cm_hemi_L[geo_dist_L < distance_threshold] = 0
        cm_hemi_R[geo_dist_R < distance_threshold] = 0
        cm_total = cm_f.copy()
        cm_total[:N_nodes_hemi_L, :N_nodes_hemi_L] = cm_hemi_L
        cm_total[N_nodes_hemi_L:, N_nodes_hemi_L:] = cm_hemi_R

        return cm_total


def assign_subject_communities_to_reference(subject_labels, target_labels, jaccard_threshold=0.1):
    """
    Assign each subject community to the best-matching target network based on Jaccard index.

    Parameters:
    - subject_labels: 1D numpy array of community labels from subject
    - target_labels: 1D numpy array of known target network labels (same length as subject_labels)
    - jaccard_threshold: minimum Jaccard score required for assignment

    Returns:
    - relabeled_subject: 1D array of same shape as subject_labels with assigned network IDs
    - assignments: list of (subject_comm, best_target_net, jaccard_score)
    """
    from sklearn.metrics import jaccard_score

    subject_labels = np.array(subject_labels)
    target_labels = np.array(target_labels)

    relabeled_subject = np.full_like(subject_labels, fill_value=-1)
    assignments = []

    subject_comms = np.unique(subject_labels)
    subject_comms = subject_comms[subject_comms != -1]  # Exclude background (0)

    target_networks = np.unique(target_labels)
    target_networks = target_networks[target_networks != -1]
    target_networks = target_networks[target_networks != 0]

    for comm in subject_comms:
        comm_mask = (subject_labels == comm)

        best_net = None
        best_score = 0

        for net in target_networks:
            target_mask = (target_labels == net)
            if target_mask.sum() == 0:
                continue

            score = jaccard_score(comm_mask, target_mask)
            if score > best_score:
                best_score = score
                best_net = net

        if best_score >= jaccard_threshold:
            relabeled_subject[comm_mask] = best_net
            assignments.append((comm, best_net, best_score))
        else:
            assignments.append((comm, -1, best_score))  # Unassigned

    return relabeled_subject, assignments

def get_consensus_assignment(relabeled_assignments, unassigned_label=-1):
    """
    Collapse assignments across thresholds (sparse → dense) into a consensus map.
    """
    n_thresholds = len(relabeled_assignments)
    n_vertices = relabeled_assignments[0].shape[0]

    # Initialize all vertices as unassigned
    consensus = np.full(n_vertices, unassigned_label)

    # Iterate from sparse to dense
    for threshold_map in relabeled_assignments:
        # Only update vertices that are still unassigned
        update_mask = (consensus == unassigned_label) & (threshold_map != unassigned_label)
        consensus[update_mask] = threshold_map[update_mask]

    return consensus

save_ind_thresh_labels = True
save_plot = True

def main(sub,bids_folder = '/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd',
        confspec = '36Pscrub3BPfilter', hemi_to_plot='R',
        ses=1, task='chase'):  
    
    #sub = f'{int(sub):02d}'
    source_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}') # .parcel
    target_folder = op.join(bids_folder,'derivatives','networks_infomap_full')
    plot_folder = op.join(bids_folder,'plots_and_ims','networks_infomap_full')
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    cm_file = op.join(source_folder,f'sub-{sub}_ses-{ses}_task-{task}_runs-9.npy')
    #import glob
    #cm_fn_pattern = op.join(source_folder, f"sub-{sub}_ses-{ses}_task-{task}_confspec-{confspec}runFD104-*runs_CM-unfiltered.npy")
    #matching_files = glob.glob(cm_fn_pattern)
    cm_f = np.load(cm_file)
    cm_filtered = spatial_filtering(cm_f) # , bids_folder=bids_folder

    #fn_target_labels_caNets = op.join(target_folder, f'sub-average_target_labels_caNets_hemi-both_thresh-0.1_prefNmod-15_confspec-36Pscrub3BPfilter.npy')
    #fn_target_labels_caNets = op.join(target_folder, f'sub-average_consensusMapping_confspec-{confspec}.npy') # mapped to ColeAnticevic nets
    target_labels_caNets = get_target_labels_caNets()  # np.load(fn_target_labels_caNets)

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

        relabeled_subject, assignments = assign_subject_communities_to_reference(full_module_mapping, target_labels_caNets,  jaccard_threshold=0.1)
        sub_module_mappings_relabelled.append(relabeled_subject)

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
        #from utils import plot_nets_CAcolors
        figure = plot_nets_CAcolors(modules_fsav5, hemi_to_plot='R')
        figure.suptitle(f'sub {sub}', y=0.75)
        plot_fn = op.join(plot_folder, f'sub-{sub}_ses-{ses}_task-{task}_networks_infomap_hemi-R_confspec-{confspec}.png')
        figure.savefig(plot_fn, dpi=300, bbox_inches='tight')
        plt.close(figure)
    print(f'sub-{sub} finished!')

if __name__ == '__main__':  
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None) #, type=int)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd', type=str, help='BIDS folder path')
    parser.add_argument('--confspec', default='36Pscrub3BPfilter', type=str, help='Configuration specification')
    parser.add_argument('--hemi_to_plot', default='R', type=str, help='Hemisphere to plot')
    parser.add_argument('--ses', default=1, type=int, help='Session number')
    parser.add_argument('--task', default='chase', type=str, help='Task name')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, confspec=args.confspec, hemi_to_plot=args.hemi_to_plot, ses=args.ses, task=args.task)

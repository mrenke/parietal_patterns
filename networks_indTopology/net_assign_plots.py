import nilearn
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
from brainspace.utils.parcellation import map_to_labels

from numrisk.fmri_analysis.gradients.utils import get_basic_mask
mask, labeling_noParcel = get_basic_mask()

hemi = 'both'  # 
import numpy as np
from sklearn.metrics import jaccard_score

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

def main(sub,bids_folder = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk',
        confspec = '36Pscrub3BPfilter',
        thresh_conn = 0.1 ,
        preferred_number_of_modules=None,
        hemi_to_plot='R',
        ses=1, task='magjudge', specification=''):  
    
    sub = f'{int(sub):02d}'
    source_folder = op.join(bids_folder,'derivatives','networks_infomap')
    plot_folder = op.join(bids_folder,'plots_and_ims','networks_infomap')

    fn_subject_modules = op.join(source_folder, f'sub-{sub}_module_mapping_infomap_hemi-{hemi}_thresh-{thresh_conn}_prefNmod-{preferred_number_of_modules}_confspec-{confspec}.npy')
    module_mapping = np.load(fn_subject_modules)

    #fn_target_labels_caNets = op.join(source_folder, f'sub-average_target_labels_caNets_hemi-{hemi}_thresh-{thresh_conn}_prefNmod-{preferred_number_of_modules}_confspec-{confspec}.npy')
    # take the best Network labellings from the group average (which is with thresh-0.1. & prefNmod-15)
    fn_target_labels_caNets = op.join(source_folder, f'sub-average_target_labels_caNets_hemi-{hemi}_thresh-0.1_prefNmod-15_confspec-{confspec}.npy')
    target_labels_caNets = np.load(fn_target_labels_caNets) # from group average 

    subject_labels = module_mapping[:, 1]  
    relabeled_subject, assignments = assign_subject_communities_to_reference(subject_labels, target_labels_caNets,  jaccard_threshold=0.1)

    modules_fsav5 = np.full(mask.shape[0], np.nan, dtype=float)
    modules_fsav5[mask] = relabeled_subject

    # Plotting
    import matplotlib.patches as mpatches
    import hcp_utils as hcp
    from matplotlib.colors import ListedColormap

    rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])
    grey = np.array([[0.5, 0.5, 0.5, 1.0]])  # RGBA format: grey with full opacity
    cmap_ca = ListedColormap( np.vstack([grey, rgb]))

    from  nilearn.datasets import fetch_surf_fsaverage
    import nilearn.plotting as nplt 
    fsaverage = fetch_surf_fsaverage('fsaverage5') 
    views = ['medial','lateral','dorsal','posterior']
    cmap = cmap_ca #'Paired'#''viridis' # 

    i_hemi_to_plot = 0 if hemi_to_plot == 'L' else 1     #hemi_to_plot = 'R'
    modules_fsav5_hemi = np.split(modules_fsav5,2)[i_hemi_to_plot]
    surf_map = modules_fsav5_hemi
    hemi_to_plot = hemi if hemi != 'both' else hemi_to_plot
    surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
    bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left

    figure, axes = plt.subplots(nrows=1, ncols=len(views),figsize = (15,8), subplot_kw=dict(projection='3d'))
    for i,view in enumerate(views):
        colbar = True if view == 'posterior' else False
        nplt.plot_surf(surf_mesh=surf_mesh , surf_map= surf_map, avg_method = 'median',# infl_right # pial_right
                view= view,cmap=cmap, colorbar=colbar, #title=f'sub-{sub}, grad {n_grad+1}',
                vmin = 0, vmax=12, # for 13 labels from caNets
                bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[i]) 
    figure.subplots_adjust(wspace=0.01)
    figure.suptitle(f'sub {sub} \n thresh {thresh_conn} \n prefNmod {preferred_number_of_modules}', y=0.75)

    fn_plot = op.join(plot_folder, f'sub-{sub}_hemi-{hemi}_thresh-{thresh_conn}_prefNmod-{preferred_number_of_modules}_confspec-{confspec}.png')
    figure.savefig(fn_plot, dpi=300, bbox_inches='tight')   
    plt.close(figure)
    print(f'Plot for sub {sub} saved to {fn_plot}')

if __name__ == '__main__':  
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk', type=str, help='BIDS folder path')
    parser.add_argument('--confspec', default='36Pscrub3BPfilter', type=str, help='Configuration specification')
    parser.add_argument('--thresh_conn', default=0.1, type=float, help='Threshold for connection matrix')
    parser.add_argument('--preferred_number_of_modules', default=None, type=int, help='Preferred number of modules for Infomap')
    parser.add_argument('--hemi_to_plot', default='R', type=str, help='Hemisphere to plot')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, confspec=args.confspec, thresh_conn=args.thresh_conn, preferred_number_of_modules=args.preferred_number_of_modules, hemi_to_plot=args.hemi_to_plot)


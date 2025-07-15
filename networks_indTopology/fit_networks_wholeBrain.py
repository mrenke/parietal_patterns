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

def threshold_matrix(mat, proportion=0.1):
    n = mat.shape[0]
    mat = mat.copy()
    np.fill_diagonal(mat, 0)     # Remove diagonal

    thresh = np.percentile(mat[mat > 0], 100 - 100 * proportion)     # Find threshold value
    mat[mat < thresh] = 0
    return mat

def main(sub, confspec = '36Pscrub3BPfilter',
        bids_folder = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk',
        thresh_conn = 0.1 ,
        ses=1, task='magjudge', specification=''):  
            
    sub = f'{int(sub):02d}'
    source_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices.tryNoHalo') # .parcel
    target_folder = op.join(bids_folder,'derivatives','networks_infomap')
    plot_folder = op.join(bids_folder,'plots_and_ims','networks_infomap')
    
    fn_path = op.join(target_folder, f'sub-{sub}_module_mapping_infomap_hemi-{hemi}_thresh-{thresh_conn}_confspec-{confspec}.npy')
    if  op.exists(fn_path):
        print(f'Module mapping file already exists: {fn_path}. Skipping to plotting')
        module_mapping = np.load(op.join(target_folder, f'sub-{sub}_module_mapping_infomap_hemi-{hemi}_thresh-{thresh_conn}_confspec-{confspec}.npy'))

    else:
        cm_file = op.join(source_folder,f'sub-{sub}_ses-1_task-magjudge_confspec-{confspec}runFD104-6runs_CM-unfiltered.npy')
        cm_f = np.load(cm_file)
        from nilearn import datasets

        # spatial exclusion: 
        # Correlations between vertices/voxels within 30 mm of each other were set to zero
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

        conn_matrix = cm_total # Proportion of connections to keep
        cm_thresh = threshold_matrix(conn_matrix, proportion= thresh_conn)
        print(sub, hemi, thresh_conn)
        from infomap import Infomap

        mat = cm_thresh
        preferred_number_of_modules = 10  # Set your preferred number of modules here

        N = mat.shape[0]
        im = Infomap(preferred_number_of_modules=preferred_number_of_modules) # add flags like '--two-level' if needed
        for i in range(N):
            for j in range(i+1, N):
                w = mat[i, j]
                if w > 0:
                    im.add_link(i, j, w)
        im.run()

        modules = [[node.node_id, node.module_id] for node in im.nodes] # #modules = {node.node_id: node.module_id for node in im.nodes}
        print(np.unique(np.array(modules)[:,1]))

        #module_mapping = np.array(modules) # np.array(modules)[:,1]
        all_nodes = np.arange(N)
        returned_nodes = np.array([node.node_id for node in im.nodes])
        returned_modules = np.array([node.module_id for node in im.nodes])

        # Fill in missing
        full_module_mapping = np.full((N,), -1, dtype=int)  # -1 means unassigned
        full_module_mapping[returned_nodes] = returned_modules

        module_mapping = np.stack([all_nodes, full_module_mapping], axis=1)
        print(module_mapping.shape)
        np.save(fn_path, module_mapping)

    modules_fsav5 = np.full(mask.shape[0], np.nan, dtype=float)
    modules_fsav5[mask] = module_mapping[:,1]
    print(modules_fsav5.shape)


    # Plot
    from  nilearn.datasets import fetch_surf_fsaverage
    import nilearn.plotting as nplt 
    fsaverage = fetch_surf_fsaverage('fsaverage5') 
    views = ['medial','lateral'] # ,'dorsal','posterior'
    cmap = 'Paired'#''viridis' # 

    figure, axes = plt.subplots(nrows=2, ncols=len(views),figsize = (8,8), subplot_kw=dict(projection='3d'))

    for i_hemi_to_plot, hemi_to_plot in enumerate(['L','R']):
        modules_fsav5_hemi = np.split(modules_fsav5,2)[i_hemi_to_plot]
        print(modules_fsav5_hemi.shape)

        map = modules_fsav5_hemi
        surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
        bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left

        for i,view in enumerate(views):
            colbar = True if view == 'posterior' else False
            nplt.plot_surf(surf_mesh=surf_mesh , surf_map= map, avg_method = 'median',# infl_right # pial_right
                    view= view,cmap=cmap, colorbar=colbar, #title=f'sub-{sub}, grad {n_grad+1}',
                    bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[i_hemi_to_plot,i]) 
        
    figure.subplots_adjust(wspace=0.01)
    figure.suptitle(f'sub {sub} \n thresh {thresh_conn}', y=0.9)
    figure.savefig(op.join(plot_folder, f'sub-{sub}_hemi-{hemi}_thresh-{thresh_conn}_confspec-{confspec}.png'), dpi=300, bbox_inches='tight')   

    plt.close(figure)
    print(f'Plot for sub {sub} saved to {op.join(plot_folder, f"sub-{sub}_hemi-{hemi}_thresh-{thresh_conn}_confspec-{confspec}.png")}')

if __name__ == '__main__':  
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--confspec', default='36Pscrub3BPfilter', type=str, help='Configuration specification')
    parser.add_argument('--thresh_conn', default=0.1, type=float, help='Threshold for connection matrix')
    args = parser.parse_args()

    main(args.subject, confspec=args.confspec, thresh_conn=args.thresh_conn)    

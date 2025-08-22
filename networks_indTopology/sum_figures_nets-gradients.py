import numpy as np
import pandas as pd
import os.path as op
import os

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt 
import hcp_utils as hcp
from matplotlib.colors import ListedColormap

# network colormap
rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])
grey = np.array([[0.5, 0.5, 0.5, 1.0]])  # RGBA format: grey with full opacity
cmap_ca = ListedColormap( np.vstack([grey, rgb]))

from numrisk.fmri_analysis.gradients.utils import get_basic_mask
mask, labeling_noParcel = get_basic_mask()


def main(sub,bids_folder = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk',
        confspec = '36Pscrub3BPfilter', hemi_to_plot='R',
        ses=1, task='magjudge'):  
    
    sub = f'{int(sub):02d}'
    grad_folder = op.join(bids_folder, 'derivatives', f'gradients.{confspec}runFD104', f'sub-{sub}')
    net_folder = op.join(bids_folder, 'derivatives', 'networks_infomap_full_01')    
    plot_folder = op.join(bids_folder,'plots_and_ims','nets-grads_summaryFigures')
    os.makedirs(plot_folder, exist_ok=True)

    gms = np.load(op.join(grad_folder,  f'sub-{sub}_g-aligned_space-fsaverag5_n10.npy'))
    nets = np.load(op.join(net_folder, f'sub-{sub}_consensusMapping_confspec-{confspec}.npy'))

    nets_fsav5 = np.full(mask.shape[0], np.nan, dtype=float)
    nets_fsav5[mask] = nets

    df = pd.DataFrame({
        'g1': gms[0, :],
        'g2': gms[1, :],
        'g3': gms[2, :],
        'nets': nets_fsav5
    })

    fig = plt.figure(constrained_layout=True, figsize=(8,10))
    fsaverage = fetch_surf_fsaverage('fsaverage5') 

    # gradient coordinate space with network coloring
    subfigs = fig.subfigures(3,1)
    axs = subfigs[0].subplots(1,3)
    sns.scatterplot(df, x='g2', y='g1', hue='nets',palette=cmap_ca,s=4, legend=False, hue_order=range(1, 13), hue_norm=plt.Normalize(0, 12), ax=axs[0], alpha=0.3) #vmin = 0, vmax=12)#
    sns.scatterplot(df, x='g3', y='g1', hue='nets',palette=cmap_ca,s=4, legend=False, hue_order=range(1, 13), hue_norm=plt.Normalize(0, 12), ax=axs[1], alpha=0.3) #vmin = 0, vmax=12)#
    sns.scatterplot(df, x='g3', y='g2', hue='nets',palette=cmap_ca,s=4, legend=False, hue_order=range(1, 13), hue_norm=plt.Normalize(0, 12), ax=axs[2], alpha=0.3) #vmin = 0, vmax=12)#
    fig.suptitle(f'sub-{sub}', fontsize=16)
    sns.despine()

    # network coloring
    net_display = 'left-right_medial-lateral'
    if net_display == 'left-right_medial-lateral':
        axs = subfigs[1].subplots(1,4, subplot_kw=dict(projection='3d'))
        modules_fsav5 = df['nets'].values
        views = ['medial','lateral']
        cmap = cmap_ca 
        for hemi_to_plot in ['L', 'R']:
            i_hemi_to_plot = 0 if hemi_to_plot == 'L' else 1  
            modules_fsav5_hemi = np.split(modules_fsav5,2)[i_hemi_to_plot]
            surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
            bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left
            for i,view in enumerate(views):
                colbar = True if view == 'posterior' else False
                nplt.plot_surf(surf_mesh=surf_mesh , surf_map= modules_fsav5_hemi, avg_method = 'median',# infl_right # pial_right
                        view= view,cmap=cmap, colorbar=colbar, #title=f'sub-{sub}, grad {n_grad+1}',
                        vmin = 0, vmax=12,
                        bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axs[i_hemi_to_plot*2 + i]) 

    ## Gradients
    axs = subfigs[2].subplots(1,3, subplot_kw=dict(projection='3d'))
    hemi_to_plot = 'R'
    i_hemi_to_plot = 0 if hemi_to_plot == 'L' else 1
    surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
    bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left  
    view = 'medial'
    for i,grad in enumerate(['g1', 'g2', 'g3']):
        map_fsav5 = df[grad].values
        map_fsav5_hemi = np.split(map_fsav5,2)[i_hemi_to_plot]
        nplt.plot_surf(surf_mesh=surf_mesh, surf_map=map_fsav5_hemi, view=view, cmap='jet', 
                    colorbar=colbar,bg_map=bg_map, bg_on_data=True,
                    darkness=0.7, axes=axs[i])

    plt.savefig(os.path.join(plot_folder, f'sub-{sub}_ses-{ses}_task-{task}_confspec-{confspec}.png'))

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

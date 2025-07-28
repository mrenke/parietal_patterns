from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt 
import numpy as np
import os.path as op

import matplotlib.patches as mpatches
import hcp_utils as hcp
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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
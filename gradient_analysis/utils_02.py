from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from utils import get_basic_mask
from nilearn import datasets
from brainspace.utils.parcellation import map_to_labels

import matplotlib.colors as colors
import matplotlib.pyplot as plt


def get_GMmargulies_cmap(skewed=True): 
    # proportion of the two colormaps, defines how much space is taken by each
    first = int((128*2)-np.round(255*(1.-0.90)))
    second = (256-first)
    first = first if skewed else second
    colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
    colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))

    # combine them and build a new colormap
    cols = np.vstack((colors2,colors3))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)
    return mymap

def get_pval_colormap():
    skewed = True
    first = int((128*2)-np.round(255*(1.-0.90)))
    second = (256-first)
    first = first if skewed else second
    colors2 = plt.cm.cool(np.linspace(0.1, .98, first))
    colors3 = plt.cm.spring(np.linspace(0.25, 1, second))

    # combine them and build a new colormap
    cols = np.vstack((colors2,colors3))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols[::-1])
    return mymap

def get_behave_bauer_params(bids_folder, dataset, unbiased=False):   
    phenotype_folder = op.join(bids_folder, 'derivatives','phenotype')
    spec = '-maps_unbiased' if unbiased else ''
    magjudge_bauer_params = pd.read_csv(op.join(phenotype_folder,f'bauer-3_sds{spec}.csv')).set_index('subject') # magjudge_bauer_params_unbiased

    magjudge_bauer_params['dataset'] = dataset
    magjudge_bauer_params.set_index('dataset', append=True, inplace=True)
    #magjudge_bauer_params.drop(columns='group', inplace=True)

    return magjudge_bauer_params
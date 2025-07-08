import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns



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


def get_glasser_parcels(base_folder='/mnt_03/diverse_neuralData/atlases_parcellations', space='fsaverage'):
    atlas_left = nib.load(op.join(base_folder,f'lh_space-{space}.HCPMMP1.gii')).agg_data()
    atlas_right =  nib.load(op.join(base_folder,f'rh_space-{space}.HCPMMP1.gii')).agg_data()

    labeling = np.concatenate([(atlas_left+1000), (atlas_right+2000)]) # unique labels for left and right!
    mask = ~np.isin(labeling, [1000,2000]) # non-cortex region (unknow and medial wall) have label 0, hence 1000 & 2000 in my variation labels L/R
    # mask.sum() == len(labeling[(labeling != 1000) & (labeling != 2000)]) 
    return mask, labeling
    
def get_glasser_CAatlas_mapping(datadir = '/mnt_03/diverse_neuralData/atlases_parcellations/ColeAnticevicNetPartition'):
    glasser_CAatlas_mapping = pd.read_csv(op.join(datadir,'cortex_parcel_network_assignments.txt'),header=None)
    glasser_CAatlas_mapping.index.name = 'glasser_parcel'
    glasser_CAatlas_mapping = glasser_CAatlas_mapping.rename({0:'ca_network'},axis=1)

    CAatlas_names = pd.read_csv(op.join(datadir,'network_label-names.csv'),index_col=0)
    CAatlas_names = CAatlas_names.set_index('Label Number')
    CAatlas_names = CAatlas_names.sort_index(level='Label Number')
    
    return glasser_CAatlas_mapping, CAatlas_names


import numpy as np

def resample_to_gaussian(connectome, mean=0.5, std=0.1):
    """
    Resample the nonzero elements of a structural connectome (DTI-based) 
    to follow a Gaussian distribution with specified mean and std.
    
    Parameters:
        connectome (np.ndarray): 2D square matrix of fiber strengths.
        mean (float): Mean of the resampled Gaussian distribution.
        std (float): Standard deviation of the resampled Gaussian distribution.
    
    Returns:
        np.ndarray: Connectome with resampled fiber strengths.
    """
    conn = connectome.copy()
    # Find the indices of the nonzero entries (assuming symmetry isn't enforced yet)
    nonzero_indices = np.nonzero(conn)
    values = conn[nonzero_indices]

    # Sort original values and generate sorted Gaussian samples
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]

    # Generate sorted Gaussian random values
    gaussian_samples = np.random.normal(loc=0.0, scale=1.0, size=len(sorted_values))
    gaussian_samples_sorted = np.sort(gaussian_samples)

    # Replace each original value with a sorted Gaussian sample
    resampled_values = np.empty_like(values)
    resampled_values[sorted_indices] = gaussian_samples_sorted

    # Rescale to desired mean and std
    resampled_values = mean + std * resampled_values

    # Place back into the matrix
    resampled_conn = np.zeros_like(conn)
    resampled_conn[nonzero_indices] = resampled_values

    return resampled_conn

def get_parcel_infos(atlas_data,atlas_affine):
    from scipy.ndimage import center_of_mass
    import nibabel as nib

    labels = np.unique(atlas_data)     # Get unique parcel labels 
    labels = labels[labels != 0]       # (excluding 0, which is usually background)

    coords = []
    for label in labels:
        mask = atlas_data == label  # Binary mask for the parcel

        com_voxel = center_of_mass(mask)  # Center of mass in voxel space
        com_mni = nib.affines.apply_affine(atlas_affine, com_voxel)  # Convert voxel indices to MNI space using affine
        coords.append(com_mni)
    coords = np.array(coords)

    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords, metric='euclidean')) 

    return coords, distances

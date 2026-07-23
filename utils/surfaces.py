import nibabel as nib
import numpy as np
import os.path as op
import nibabel as nib

def get_NPC_mask(bids_folder_orig = '/mnt_03/ds-dnumrisk', space = 'fsaverage5', hemi='both'):
    surf_mask_L = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_L_space-{space}_hemi-lh.label.gii')
    surf_mask_L = nib.load(surf_mask_L).agg_data()
    surf_mask_R = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_R_space-{space}_hemi-rh.label.gii')
    surf_mask_R = nib.load(surf_mask_R).agg_data()
    if hemi == 'both':
        nprf_r2 = np.concatenate((surf_mask_L, surf_mask_R))
    if hemi == 'L':
        nprf_r2 = np.concatenate((surf_mask_L, np.zeros_like(surf_mask_R)))
    if hemi == 'R':
        nprf_r2 = np.concatenate((np.zeros_like(surf_mask_L), surf_mask_R)) 

    nprf_r2 = np.bool_(nprf_r2)
    return nprf_r2

from nilearn import datasets
def get_basic_mask():
    atlas = datasets.fetch_atlas_surf_destrieux()
    regions = atlas['labels'].copy()
    masked_regions = [b'Medial_wall', b'Unknown']
    masked_labels = [regions.index(r) for r in masked_regions]
    for r in masked_regions:
        regions.remove(r)
    labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
    labeling_noParcel = np.arange(0,len(labeling),1,dtype = int)     # Map gradients to original parcels
    mask = ~np.isin(labeling, masked_labels)
    return mask, labeling_noParcel



# ── Precompute adjacency once (shared across subjects) ─────────────────────
def build_adj_and_coords(surf_path):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import cdist

    surf = nib.load(surf_path)
    coords = surf.darrays[0].data          # (32492, 3)
    faces  = surf.darrays[1].data          # (n_faces, 3)
    n = coords.shape[0]
    i = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    j = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
    adj = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return adj, coords

def load_gifti_mesh(surf_path):
    """Load a GIFTI surface (.surf.gii) as a pyvista PolyData mesh."""
    import pyvista as pv

    surf = nib.load(surf_path)
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data
    faces_padded = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    )
    return pv.PolyData(coords, faces_padded)


def plot_surf_patch_pyvista(surf_path, surf_map, vmin, vmax, cmap='viridis',
                             azimuth=0, elevation=0, roll=None, title=None,
                             window_size=(800, 800), screenshot=None):
    """Render only the finite-valued patch of a surface map (true sub-mesh, not just
    greyed-out background) with an arbitrary camera angle, via pyvista.

    `surf_map` holds one scalar per mesh vertex, in vertex order; NaN entries are
    excluded from the mesh entirely (same convention as the surf_map_hemi arrays
    already built for nplt.plot_surf elsewhere in this notebook). Requires an X
    server (real or Xvfb) — set the DISPLAY env var before calling.
    """
    import pyvista as pv

    mesh = load_gifti_mesh(surf_path)
    mask = np.isfinite(surf_map)
    mesh['value'] = surf_map
    patch = mesh.extract_points(mask, adjacent_cells=True)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(patch, scalars='value', cmap=cmap, clim=(vmin, vmax),
                      show_scalar_bar=True, scalar_bar_args={'title': title or ''})
    plotter.background_color = 'white'
    plotter.reset_camera()
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    if roll is not None:
        plotter.camera.roll = roll
    img = plotter.screenshot(screenshot, return_img=True)
    plotter.close()
    return img


def centroid_to_lobe(centroid):
    """Approximate anatomical lobe label from patch centroid (fsLR MNI-like coords)."""
    x, y, z = centroid
    ax = abs(x)
    if y < -30 and z > 20:
        return 'parietal-lateral'
    if ax < 20 and z > 30:
        return 'frontal-medial-dorsal'
    if y > -20 and z > 15 and ax > 15:
        return 'frontal-lateral'
    if z < 5 and ax > 30 and y < -10:
        return 'temporal'
    if y > 30 and z < 20:
        return 'frontal-medial'
    return 'frontal-insula'


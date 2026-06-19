import numpy as np
from matplotlib.colors import ListedColormap

def get_CANets_cmap():
    import matplotlib.patches as mpatches
    import hcp_utils as hcp

    rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])
    grey = np.array([[0.5, 0.5, 0.5, 1.0]])  # RGBA format: grey with full opacity
    cmap_ca = ListedColormap( np.vstack([grey, rgb])) # Add grey color at the beginning

    return cmap_ca

def get_gordon17_cmap():
    """Colormap for Gordon 2017 networks, extracted from Figure 3 legend."""
    # Order matches EG17 network_names file (labels 1–17), index 0 = unassigned
    # Label 7 (Language) = "Vent Attn" in the paper figure legend
    GORDON17_COLORS = [
        (0.50, 0.50, 0.50, 1.0),  # 0  — unassigned   (grey)
        (0.90, 0.10, 0.10, 1.0),  # 1  — Default       (red)
        "#17138E" ,  # 2  — LatVis        (sky blue)
        (1.00, 0.90, 0.00, 1.0),  # 3  — FrontPar      (yellow)
        (1.00, 0.70, 0.45, 1.0),  # 4  — MedVis        (peach)
        "#1EC51B" ,  # 5  — DorsAttn      (dark green)
        (0.70, 0.50, 0.90, 1.0),  # 6  — Premotor      (lavender)
        "#468E79",  # 7  — Language      (teal)
        (0.00, 0.00, 0.00, 1.0),  # 8  — Salience      (black)
        (0.50, 0.00, 0.70, 1.0),  # 9  — CingOperc     (dark purple)
        (0.00, 0.90, 0.85, 1.0),  # 10 — HandSM        (cyan)
         "#FEA02D" ,  # 11 — FaceSM        (orange)
        (1.00, 0.60, 0.80, 1.0),  # 12 — Auditory      (pink)
        (0.30, 0.50, 0.90, 1.0),  # 13 — AntMTL        (medium blue)
        (0.50, 0.90, 0.40, 1.0),  # 14 — PostMTL       (light green)
        (0.15, 0.20, 0.60, 1.0),  # 15 — ParMemory     (navy)
        (0.95, 0.95, 0.95, 1.0),  # 16 — Context       (near white)
        (0.00, 0.50, 0.45, 1.0),  # 17 — FootSM        (dark teal)
    ]

    network_names = ['unassigned', 'Default', 'LatVis', 'FrontPar', 'MedVis',
                     'DorsAttn', 'Premotor', 'Language', 'Salience', 'CingOperc',
                     'HandSM', 'FaceSM', 'Auditory', 'AntMTL', 'PostMTL',
                     'ParMemory', 'Context', 'FootSM']

    cmap_gordon = ListedColormap(GORDON17_COLORS)
    return cmap_gordon, network_names

import nibabel as nib
import os.path as op

def get_template_vertex_area(NEUROMAPS_FSLR):
    # Medial-wall mask — tells us which of the 32492 L/R vertices are valid cortex
    # (excludes medial wall; produces 29696 L + 29716 R = 59412 nodes total)
    mw_L = nib.load(op.join(NEUROMAPS_FSLR, 'tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii'))
    mw_R = nib.load(op.join(NEUROMAPS_FSLR, 'tpl-fsLR_den-32k_hemi-R_desc-nomedialwall_dparc.label.gii'))
    cortex_mask_L = mw_L.darrays[0].data.astype(bool)   # (32492,)
    cortex_mask_R = mw_R.darrays[0].data.astype(bool)   # (32492,)

    # Template vertex areas (group-average fsLR32k)
    tpl_va_L = nib.load(op.join(NEUROMAPS_FSLR, 'tpl-fsLR_den-32k_hemi-L_desc-vaavg_midthickness.shape.gii')).darrays[0].data  # (32492,)
    tpl_va_R = nib.load(op.join(NEUROMAPS_FSLR, 'tpl-fsLR_den-32k_hemi-R_desc-vaavg_midthickness.shape.gii')).darrays[0].data

    # Concatenated template areas for valid cortical nodes only (59412,)
    tpl_vertex_areas = np.concatenate([tpl_va_L[cortex_mask_L], tpl_va_R[cortex_mask_R]])

    print(f'Valid cortical nodes: L={cortex_mask_L.sum()}, R={cortex_mask_R.sum()}, total={tpl_vertex_areas.shape[0]}')
    print(f'Total template cortical area: {tpl_vertex_areas.sum():.0f} mm² ({tpl_vertex_areas.sum()/100:.1f} cm²)')
    return tpl_vertex_areas, cortex_mask_L, cortex_mask_R

def load_individual_vertex_areas(PFM_ROOT, SUBJECTS, NEUROMAPS_FSLR=  '/home/ubuntu/neuromaps-data/atlases/fsLR'):
    _, cortex_mask_L, cortex_mask_R = get_template_vertex_area(NEUROMAPS_FSLR)

    ind_areas_dict = {}
    for sub_id in SUBJECTS:
        sub_str = f'sub-{sub_id:02d}'
        areas = []
        ok = True
        for hemi, mask in [('L', cortex_mask_L), ('R', cortex_mask_R)]:
            path = op.join(PFM_ROOT, sub_str, 'anat',
                        f'{sub_str}_ses-1_hemi-{hemi}_vertex_areas_fsLR32k.shape.gii')
            if not op.exists(path):
                print(f'  sub-{sub_id:02d}: missing hemi-{hemi} — run 06_comp_indVertexArea.py')
                ok = False
                break
            va_full = nib.load(path).darrays[0].data  # (32492,)
            areas.append(va_full[mask])               # keep valid cortex only
        if ok:
            ind_areas_dict[sub_id] = np.concatenate(areas)  # (59412,)

    print(f'Individual areas loaded for {len(ind_areas_dict)} subjects')
    return ind_areas_dict


def array_to_gifti(arr):
    darray = nib.gifti.GiftiDataArray(
        data=arr.astype(np.float32),
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
        datatype='NIFTI_TYPE_FLOAT32'
    )
    return nib.gifti.GiftiImage(darrays=[darray])


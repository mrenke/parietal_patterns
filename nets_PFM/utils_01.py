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

"""Shared plotting utilities for the DD paper neural figures.

House style follows the vision-science 'scientific-figures' convention:
seaborn-on-matplotlib, Helvetica, despined/offset axes, no gridlines,
hand-picked muted categorical palette, vector-safe export.
"""
import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

GROUP_LABELS = {0: 'Control', 1: 'Dyscalculia'}
GROUP_ORDER = ['Control', 'Dyscalculia']
GROUP_PALETTE = {'Control': '#4C72B0', 'Dyscalculia': '#DD8452'}  # blue / orange, matches other paper figures


def set_style():
    """Apply the house rcParams (Helvetica, despined/offset axes, vector-safe export)."""
    mpl.rcParams.update({
        # 'font.family' must be the *generic* family ('sans-serif'), not a literal name --
        # matplotlib only consults the font.sans-serif fallback list in that case. Pinning
        # it directly to 'Helvetica' skips the fallback and always warns since there's no
        # true Helvetica/Arial on this machine; 'Nimbus Sans' (URW) is the metric-compatible
        # clone actually installed (fc-list), so it's what this resolves to.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Nimbus Sans', 'Arial'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'mathtext.fontset': 'stixsans',

        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 4,

        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,

        'legend.frameon': False,

        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',

        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })
    sns.set_context('paper')


def format_pvalue(p, prefix='p'):
    """'p < 0.001' below the reporting floor, otherwise 'p = 0.036'.

    Used for both frequentist p-values and Bayesian posterior tail probabilities
    (pass prefix='p_bayes' for the latter). A tail probability of exactly 0 just
    means no posterior draw fell on that side, so it must never be printed as
    'p = 0.000' -- the floor is what the draw count can actually resolve.
    """
    return f'{prefix} < 0.001' if p < 0.001 else f'{prefix} = {p:.3f}'


def render_surf_panel(surf_mesh, bg_map, surf_map_hemi, view, cmap, vmin, vmax,
                       hemi='left', avg_method='median', darkness=0.3):
    """Render one inflated-surface panel to a whitespace-cropped PIL image.

    hemi: passed through to nilearn's plot_surf -- only matters when `view` is a
    named string (e.g. 'lateral'); ignored when `view` is an explicit (elev, azim)
    tuple, which overrides the camera regardless of hemi.
    """
    import nilearn.plotting as nplt

    fig_panel = plt.figure(figsize=(4, 4))
    ax_panel = fig_panel.add_subplot(111, projection='3d')
    nplt.plot_surf(surf_mesh=surf_mesh, surf_map=surf_map_hemi, avg_method=avg_method,
                    hemi=hemi, view=view, cmap=cmap, colorbar=False,
                    vmin=vmin, vmax=vmax, bg_map=bg_map, bg_on_data=True, darkness=darkness,
                    axes=ax_panel, figure=fig_panel)
    buf = io.BytesIO()
    fig_panel.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig_panel)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    # auto-crop whitespace so panels pack tightly in the grid
    arr = np.array(img)
    nonwhite = np.any(arr < 250, axis=-1)
    r0, r1 = np.where(np.any(nonwhite, axis=1))[0][[0, -1]]
    c0, c1 = np.where(np.any(nonwhite, axis=0))[0][[0, -1]]
    pad = 5
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, arr.shape[0]), min(c1 + pad, arr.shape[1])
    return img.crop((c0, r0, c1, r1))


def add_panel_letter(ax, letter, dx=-0.15, dy=1.05, fontsize=12):
    """Nature-style lowercase bold panel letter, top-left, outside the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='bottom', ha='right')


def add_panel_letter_fig(fig, letter, x, y, fontsize=12):
    """Panel letter placed in *figure* coordinates, for a shared letter baseline.

    add_panel_letter() offsets from an axes, which drifts as soon as the panels it
    labels differ in size, aspect, or how far their tick/axis labels stick out -- the
    letters then land at visibly different heights. Pass figure coordinates (e.g. from
    a GridSpec cell's get_position(fig)) when several letters must line up.
    """
    fig.text(x, y, letter, fontsize=fontsize, fontweight='bold', va='bottom', ha='left')


def add_subpanel_label(ax, label, dx=-0.15, dy=1.05, fontsize=9):
    """Small italic sub-panel index (i, ii, iii, ...) for panels grouped under one main letter."""
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=fontsize,
            fontstyle='italic', va='bottom', ha='right', color='0.2')


def plot_group_comparison_panel(ax, df_plot, measure, ylabel, stat,
                                 order=GROUP_ORDER, palette=GROUP_PALETTE, ymin=None):
    """One panel: bar (alpha=0.5) + swarm overlay + a significance bracket with the p-value.

    Matches the bar+swarm idiom already used for group comparisons elsewhere in
    this paper (gradients_noHalo/rep_groupDiffs_dParams.ipynb), kept here as the
    house convention rather than switching to a different mark type per figure.

    ymin: optional zoomed-in lower y-limit (bars still drawn from 0, only the
    view is cropped) for a measure whose distribution is hard to see against a
    from-zero axis.
    """
    sns.barplot(data=df_plot, x='group_label', y=measure, hue='group_label',
                order=order, hue_order=order, palette=palette, alpha=0.5,
                legend=False, ax=ax)
    sns.swarmplot(data=df_plot, x='group_label', y=measure, hue='group_label',
                  order=order, hue_order=order, palette=palette, size=4,
                  legend=False, ax=ax)

    # significance bracket spanning the two bars (x=0, x=1), short legs that stay
    # well clear of the bars/swarm, with just the p-value (no test statistic)
    y_span = df_plot[measure].max() - df_plot[measure].min()
    bracket_y = df_plot[measure].max() + 0.08 * y_span
    leg = 0.03 * y_span
    ax.plot([0, 0, 1, 1], [bracket_y - leg, bracket_y, bracket_y, bracket_y - leg],
            lw=0.8, color='0.15')

    ax.text(0.5, bracket_y, format_pvalue(stat.pvalue), ha='center', va='bottom',
            fontsize=7, color='0.15')

    ax.set_ylim(top=bracket_y + 0.18 * y_span)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)

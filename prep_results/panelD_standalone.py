"""Standalone panel D -- exported on its own to be swapped into the Affinity document."""
import os
import os.path as op
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, '/Users/mrenke/git')
from parietal_patterns.prep_results.plotting import (
    set_style, format_pvalue, GROUP_ORDER, GROUP_PALETTE,
)

set_style()

BIDS = '/Users/mrenke/data/ds-dnumrisk'
COGMODELS = op.join(BIDS, 'derivatives', 'cogmodels_magjudge')
OUT_DIRS = ['/Users/mrenke/Desktop/DNumRisk/figures/paperFigs_DD_03',
            '/Users/mrenke/obsidian-wiki/dyscalc_paper_rework/figures/plots']
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

rdm = az.from_netcdf(op.join(COGMODELS, 'model-rdm_full_cont_hn_trace.nc'))


def softplus(x):
    """bauer.utils.bayes.softplus, inlined (importing it pulls in pytensor)."""
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def panel_D_standalone():
    params = ['perceptual_noise_sd', 'memory_noise_sd', 'a']
    names = ['Perceptual noise', 'Memory noise', 'Boundary separation ($a$)']

    # 3 params + a legend column on the right, matching where the legend sat in the
    # old panel D so it drops into the same slot in the Affinity document
    fig, axes = plt.subplots(1, 4, figsize=(5.6, 1.15),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.55], 'wspace': 0.3})
    ax_leg = axes[3]

    for ax, param, name in zip(axes[:3], params, names):
        tmp = (rdm.posterior[f'{param}_mu'].to_dataframe()[f'{param}_mu']
               .unstack(f'{param}_regressors'))
        by_group = {'Control': softplus(tmp['C(group)[control]']),
                    'Dyscalculia': softplus(tmp['C(group)[dyscalc]'])}
        # two-sided posterior tail probability of the group difference
        below = float(np.mean((by_group['Dyscalculia'] - by_group['Control']) < 0))
        p = below if below < 0.5 else 1 - below

        for label in GROUP_ORDER:
            sns.kdeplot(x=by_group[label], color=GROUP_PALETTE[label], fill=True,
                        lw=1.0, alpha=0.45, legend=False, ax=ax)
        ax.set(yticks=[], xlabel='')
        ax.set_ylabel('Density' if ax is axes[0] else '', fontsize=8)
        # title and p-value are separate texts rather than one two-line title, so the gap
        # between them (pad vs. the p-value's y) is tunable independently of line spacing
        ax.set_title(name, fontsize=8, pad=17)
        # p-value as a lighter second line, so the parameter name stays the headline
        ax.text(0.5, 1.01, format_pvalue(p, 'p$_{bayes}$'), transform=ax.transAxes,
                ha='center', va='bottom', fontsize=7, color='0.35')
        ax.locator_params(axis='x', nbins=3)
        ax.margins(x=0.08)
        sns.despine(ax=ax, left=True, trim=False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_PALETTE[g], alpha=0.45,
                              edgecolor=GROUP_PALETTE[g], lw=1.0, label=g)
               for g in GROUP_ORDER]
    ax_leg.legend(handles=handles, loc='center left', handlelength=1.1,
                  handleheight=1.0, borderpad=0.0, labelspacing=0.5, fontsize=8)
    ax_leg.axis('off')
    return fig


if __name__ == '__main__':
    fig = panel_D_standalone()
    for d in OUT_DIRS:
        fig.savefig(op.join(d, 'panelD_mag_bauerv4_rdm_full_cont_control-dys.pdf'))
    fig.savefig('/private/tmp/claude-479260791/-Users-mrenke-git-parietal-patterns/'
                '3eab62e7-4cc9-452b-8e59-fbb8116887a6/scratchpad/panelD_preview.png', dpi=300)
    print('done')

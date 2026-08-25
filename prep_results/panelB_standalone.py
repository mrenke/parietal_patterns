"""Standalone panel B -- exported on its own to be swapped into the Affinity document.

Reworked from the old panel B: the two grey probit-posterior densities are dropped (they
only restated a number that now sits over the curve it belongs to), and the freed space
goes to raw behaviour as the bar+swarm idiom used in the neural figures.

    row 1   accuracy, reaction time   -- plot_group_comparison_panel (bar + swarm + bracket)
    row 2   psychophysical curves, number-range interaction, each carrying the
            corresponding probit group-interaction p_bayes as a subtitle

Run with mambaforge/envs/behav_fit -- plot_group_comparison_panel needs seaborn >= 0.13
(`legend=` on barplot), unlike panelD_standalone.py which runs anywhere.
"""
import os
import os.path as op
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu, normaltest, ttest_ind

sys.path.insert(0, '/Users/mrenke/git')
from parietal_patterns.prep_results.plotting import (
    set_style, format_pvalue, plot_group_comparison_panel,
    GROUP_LABELS, GROUP_ORDER, GROUP_PALETTE,
)

set_style()

BIDS = '/Users/mrenke/data/ds-dnumrisk'
COGMODELS = op.join(BIDS, 'derivatives', 'cogmodels_magjudge')
OUT_DIRS = ['/Users/mrenke/Desktop/DNumRisk/figures/paperFigs_DD_03',
            '/Users/mrenke/obsidian-wiki/dyscalc_paper_rework/figures/plots']
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

ALPHA = 0.05
N_BOOT = 1000  # bootstrap reps for the logistic-fit CI band

from numrisk.behavior_magjudge.utils import get_data

df = get_data(BIDS, include_var=['group']).sort_index()
df['group_label'] = df['group'].map(GROUP_LABELS)
df['n1'] = df['n1'].astype(int)
df['x'] = df['log(n2/n1)']

# subject-wise accuracy and RT (same derivation as numrisk paper_dnumrisk_01.ipynb cell 4)
df['correct'] = (df['n2'] > df['n1']) == df['chose_n2']
subj = (df.groupby(['subject', 'group_label'])[['correct', 'rt']].mean().reset_index())

probit = az.from_netcdf(op.join(COGMODELS, 'probit_model-2_trace.netcdf'))


# directional a-priori hypothesis per measure, phrased as the alternative for
# (control vs dyscalculia): controls more accurate, controls faster. We never
# expected dyscalculics to be better or faster, so these are tested one-sided.
ALTERNATIVE = {'correct': 'greater', 'rt': 'less'}


def group_stat(measure):
    """t-test or Mann-Whitney depending on normality -- the rule already used for these
    two measures in numrisk paper_dnumrisk_01.ipynb, kept so the numbers don't move.

    One-sided in the direction of ALTERNATIVE[measure]: the hypothesis is that
    dyscalculics are less accurate and slower, and the opposite direction was
    neither expected nor of interest."""
    ctrl = subj.loc[subj['group_label'] == 'Control', measure]
    dys = subj.loc[subj['group_label'] == 'Dyscalculia', measure]
    _, p_normal = normaltest(subj[measure].dropna())
    if p_normal > ALPHA:
        return ttest_ind(ctrl, dys, alternative=ALTERNATIVE[measure])
    return mannwhitneyu(ctrl, dys, alternative=ALTERNATIVE[measure])


def probit_p(regressor):
    """Smaller posterior tail of a probit group-interaction term."""
    samples = probit.posterior[regressor].to_dataframe()[regressor]
    above = float(np.mean(samples > 0))
    return min(above, 1 - above)


def subtitle(ax, name, note):
    """Panel name + a lighter p-value line -- same two-text arrangement as panel D, so the
    gap between them stays tunable independently of title line spacing."""
    ax.set_title(name, fontsize=8, pad=17)
    ax.text(0.5, 1.01, note, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=6.5, color='0.35')


def panel_B_standalone():
    fig, axes = plt.subplots(2, 2, figsize=(4.35, 3.7))
    (ax_acc, ax_rt), (ax_curve, ax_range) = axes
    fig.subplots_adjust(wspace=0.42, hspace=0.90, left=0.13, right=0.99,
                        top=0.93, bottom=0.11)

    # ── row 1: raw behaviour, house bar+swarm idiom ──
    plot_group_comparison_panel(ax_acc, subj, 'correct', 'Prop. correct',
                                group_stat('correct'), ymin=0.5)
    plot_group_comparison_panel(ax_rt, subj, 'rt', 'Reaction time (s)',
                                group_stat('rt'))
    for ax in (ax_acc, ax_rt):
        ax.tick_params(axis='x', labelsize=7)

    # ── row 2: choice curves, each carrying its probit interaction p_bayes ──
    for label in GROUP_ORDER:
        sub = df[df['group_label'] == label]
        sns.regplot(data=sub, x='x', y='chose_n2', logistic=True, scatter=False,
                    ci=95, n_boot=N_BOOT, color=GROUP_PALETTE[label],
                    line_kws={'linewidth': 1.4}, ax=ax_curve)
    ax_curve.axvline(0, color='0.6', ls='--', lw=0.8)
    ax_curve.axhline(0.5, color='0.6', ls='--', lw=0.8)
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_yticks([0, 0.5, 1])
    ax_curve.set_xlabel('log(n2/n1)')
    ax_curve.set_ylabel('Prop. chose n2')
    handles = [plt.Line2D([], [], color=GROUP_PALETTE[g], lw=1.4, label=g)
               for g in GROUP_ORDER]
    ax_curve.legend(handles=handles, loc='upper left', fontsize=6.5, handlelength=1.1,
                    borderpad=0.15, labelspacing=0.2, borderaxespad=0.2)
    subtitle(ax_curve, 'Psychophysical curves',
             f'slope $\\times$ group: {format_pvalue(probit_p("x:group"), "p$_{bayes}$")}')
    sns.despine(ax=ax_curve)

    sns.pointplot(data=df.reset_index(), x='n1', y='chose_n2', hue='group_label',
                  hue_order=GROUP_ORDER, palette=GROUP_PALETTE, dodge=0.25,
                  errorbar=('ci', 95), markersize=3, linewidth=1.2,
                  err_kws={'linewidth': 1.0}, ax=ax_range)
    ax_range.axhline(0.5, color='0.6', ls='--', lw=0.8)
    ax_range.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax_range.set_xlabel('Number range (n1)')
    ax_range.set_ylabel('Prop. chose n2')
    ax_range.legend_.remove()  # shared with the curve panel
    subtitle(ax_range, 'Number range interaction',
             f'range $\\times$ group: {format_pvalue(probit_p("n1:group"), "p$_{bayes}$")}')
    sns.despine(ax=ax_range)

    return fig


if __name__ == '__main__':
    fig = panel_B_standalone()
    for d in OUT_DIRS:
        fig.savefig(op.join(d, 'panelB_mag_behav_probit_control-dys.pdf'))
    fig.savefig('/private/tmp/claude-479260791/-Users-mrenke-git-parietal-patterns/'
                '3eab62e7-4cc9-452b-8e59-fbb8116887a6/scratchpad/panelB_preview.png', dpi=300)
    print('written to', ', '.join(OUT_DIRS))

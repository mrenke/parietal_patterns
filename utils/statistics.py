from scipy.stats import normaltest, ttest_ind, mannwhitneyu, ttest_rel, sem, t
import numpy as np
import pandas as pd

def group_summary_table(df_tmp, measures, group_names=[0, 1], group_labels=None, ci=0.95):
    """Descriptive stats (N, mean, SD, CI of the mean, median, IQR) per group, per measure.

    df_tmp must have 'group' either as a column or as a level in the index (as produced
    by between_group_comparison's caller). One row per (measure, group) combination.
    """
    if 'group' not in df_tmp.columns:
        df_tmp = df_tmp.reset_index('group')

    rows = []
    for m in measures:
        for g in group_names:
            values = df_tmp.loc[df_tmp['group'] == g, m].dropna()
            n = len(values)
            mean = values.mean()
            sd = values.std(ddof=1)
            se = sem(values)
            ci_halfwidth = se * t.ppf((1 + ci) / 2, n - 1) if n > 1 else np.nan
            q1, q3 = values.quantile([0.25, 0.75])
            rows.append({
                'measure': m,
                'group': group_labels[g] if group_labels is not None else g,
                'N': n,
                'mean': mean,
                'sd': sd,
                f'ci{int(ci*100)}_low': mean - ci_halfwidth,
                f'ci{int(ci*100)}_high': mean + ci_halfwidth,
                'median': values.median(),
                'iqr_low': q1,
                'iqr_high': q3,
            })

    return pd.DataFrame(rows).set_index(['measure', 'group'])


def compact_summary_table(df_tmp, measures, stats_by_measure, group_names=[0, 1], group_labels=None,
                           unit='', decimals=2):
    """Paper-ready compact summary table: one row per measure, columns
    '{group} ({unit} ± SD)' (x2, one per group), 'Statistic', 'p (uncorr)'.

    House format for all paper-ready summary tables in prep_results/ -- see
    prep_results/CLAUDE.md. `unit` applies to every measure in the table, so
    call this once per unit (e.g. once for R²/vertex-count measures with
    unit='', once for cm² area measures with unit='cm²') rather than mixing
    units within a single call.

    stats_by_measure: {measure: (stat, stats_term)}, as produced by looping
    between_group_comparison() over `measures`.
    decimals: int, or {measure: int} for per-measure precision.
    """
    if 'group' not in df_tmp.columns:
        df_tmp = df_tmp.reset_index('group')

    if not isinstance(decimals, dict):
        decimals = {m: decimals for m in measures}

    g0, g1 = group_names
    label0 = group_labels[g0] if group_labels is not None else g0
    label1 = group_labels[g1] if group_labels is not None else g1
    col0 = f'{label0} ({unit} ± SD)' if unit else f'{label0} (mean ± SD)'
    col1 = f'{label1} ({unit} ± SD)' if unit else f'{label1} (mean ± SD)'

    def fmt_p(p):
        return '<0.001' if p < 0.0005 else f'{p:.3f}'

    rows = []
    for m in measures:
        d = decimals[m]
        v0 = df_tmp.loc[df_tmp['group'] == g0, m].dropna()
        v1 = df_tmp.loc[df_tmp['group'] == g1, m].dropna()
        stat, stats_term = stats_by_measure[m]
        rows.append({
            'measure': m,
            col0: f'{v0.mean():.{d}f} ± {v0.std(ddof=1):.{d}f}',
            col1: f'{v1.mean():.{d}f} ± {v1.std(ddof=1):.{d}f}',
            'Statistic': f'{stats_term} = {stat.statistic:.2f}',
            'p (uncorr)': fmt_p(stat.pvalue),
        })

    return pd.DataFrame(rows).set_index('measure')


def between_group_comparison(df_tmp, y_var, alpha=0.05, group_names = ['Control','Dyscalculic']):
    pval_normal = normaltest(df_tmp[y_var]).pvalue
    if 'group' not in df_tmp.columns:
           df_tmp = df_tmp.reset_index('group')

    group1 = df_tmp[df_tmp['group'] == group_names[0]][y_var].dropna()
    group2 = df_tmp[df_tmp['group'] == group_names[1]][y_var].dropna()

    if pval_normal > alpha:
            stats = ttest_ind(group1, group2, axis=0)
            stats_term = 't'
    else: # non parametric test
            stats = mannwhitneyu(group1, group2, axis=0)
            stats_term = 'U'
            
    return stats, stats_term



def get_pval_colormap():
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt

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

def sig_stars(p):
    if pd.isna(p):
        return ''
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''
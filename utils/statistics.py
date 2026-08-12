from scipy.stats import normaltest, ttest_ind, mannwhitneyu, ttest_rel
import numpy as np
import pandas as pd

def between_group_comparison(df_tmp, y_var, alpha=0.05, group_names = ['Control','Dyscalculic']):
    pval_normal = normaltest(df_tmp[y_var]).pvalue
    if 'group' not in df_tmp.columns:
           df_tmp = df_tmp.reset_index('group')

    group1 = df_tmp[df_tmp['group'] == group_names[0]][y_var].dropna()
    group2 = df_tmp[df_tmp['group'] == group_names[1]][y_var].dropna()

    if pval_normal > alpha:
            stats = ttest_ind(group1, group2, axis=0)
            stats_term = f't({len(group1)+len(group2)-2})'
    else: # non parametric test
            stats = mannwhitneyu(group1, group2, axis=0)
            stats_term = f'U({len(group1)}, {len(group2)})'
            
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
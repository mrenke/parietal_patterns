from scipy.stats import normaltest, ttest_ind, mannwhitneyu, ttest_rel

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
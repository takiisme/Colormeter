import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots.bundles import icml2024
from tueplots.constants.color import rgb
from constants import GT

nrows, ncols = 1, 2
plt.rcParams.update(icml2024(column='full', nrows=nrows, ncols=ncols))


# stats_acc = pd.read_csv('cv_k_out_detailed.csv', index_col=0).assign(
#     below2 = lambda x: x['delta_E'] < 2.0,
#     below5 = lambda x: x['delta_E'] < 5.0,
#     below10 = lambda x: x['delta_E'] < 10.0,
# ).groupby(['k']).agg(
#     below2_rate = ('below2', 'mean'),
#     below5_rate = ('below5', 'mean'),
#     below10_rate = ('below10', 'mean')
# )#.reset_index().groupby('k').agg(
# #     below2_rate = ('below2_rate', 'mean'),
# #     below2_rate_std = ('below2_rate', 'std'),
# #     below5_rate = ('below5_rate', 'mean'),
# #     below5_rate_std = ('below5_rate', 'std'),
# #     below10_rate = ('below10_rate', 'mean'),
# #     below10_rate_std = ('below10_rate', 'std')
# # )
# # print(results_k)
# stats_mse = pd.read_csv('cv_k_out_detailed.csv', index_col=0).groupby(['k', 'iteration']).agg(
#     avg_error = ('delta_E', 'mean')
# ).reset_index().groupby('k').agg(
#     avg_error = ('avg_error', 'mean'),
#     avg_error_std = ('avg_error', 'std')
# ).reset_index()

cv_stats = pd.read_csv('cv_k_out_detailed.csv', index_col=0).assign(
    below5 = lambda x: x['delta_E'] < 5.0
).groupby(['k', 'iteration']).agg(
    below5_rate = ('below5', 'mean'),
    avg_error = ('delta_E', 'mean')
).groupby('k').agg(
    below5_rate = ('below5_rate', 'mean'),
    below5_rate_lq = ('below5_rate', lambda x: np.percentile(x, 25)),
    below5_rate_uq = ('below5_rate', lambda x: np.percentile(x, 75)),
    avg_error = ('avg_error', 'mean'),
    avg_error_std = ('avg_error', 'std'),
    scaling_avg_error_lq = ('avg_error', lambda x: np.percentile(x, 25)),
    scaling_avg_error_uq = ('avg_error', lambda x: np.percentile(x, 75)),
).reset_index()
# print(cv_stats.sort_values(by=['below5_rate'], ascending=False))
# exit()

df_test = pd.read_csv('Data/test_corrected.csv').assign(
    scaling_delta_E = lambda df: np.sqrt(
        (df['scaling_r4_l'] - df['gt__l']) ** 2 +
        (df['scaling_r4_a'] - df['gt__a']) ** 2 +
        (df['scaling_r4_b'] - df['gt__b']) ** 2
    )
)
scaling_below5 = np.mean(df_test['scaling_delta_E'] < 5.0)
# scaling_below10 = np.mean(df_test['scaling_delta_E'] < 10.0)
scaling_avg_error = np.mean(df_test['scaling_delta_E'])
scaling_avg_error_std = np.std(df_test['scaling_delta_E'])


fig, axs = plt.subplots(nrows, ncols)
# < 5 error rate
ax = axs[0]
# ax.errorbar(
#     cv_stats['k'],
#     cv_stats['below5_rate'],
#     yerr=[cv_stats['below5_rate'] - cv_stats['below5_rate_lq'], cv_stats['below5_rate_uq'] - cv_stats['below5_rate']],
#     linestyle='dotted', marker='o',
#     # label=r'Reduced model',
#     color=rgb.tue_green,
#     capsize=3, markersize=3
# )
ax.plot(
    cv_stats['k'],
    cv_stats['below5_rate'],
    linestyle='-', marker='o',
    # label=r'Reduced model',
    color=rgb.tue_green,
    markersize=3
)
ax.fill_between(
    cv_stats['k'],
    cv_stats['below5_rate_lq'],
    cv_stats['below5_rate_uq'],
    color=rgb.tue_green, alpha=0.2
)
ax.axhline(
    y=scaling_below5,
    color=rgb.tue_green,
    linestyle='-', alpha=0.5,
    label='Scaling, Euclidean error $<5$'
)
ax.set_ylim(0.00, 0.30)
yticks = np.arange(0.00, 0.35, 0.05)
ax.set_yticks(
    ticks=yticks,
    # labels=[f'{y*100:.0f}\\%' for y in yticks]
)
ax.set_ylabel(r'Proportion of Euclidean error $<5$', color=rgb.tue_green)
xticks = np.arange(2, 22, 2)
ax.set_xticks(xticks)
# ax.legend()

# Euclidean error
ax2 = ax.twinx()
# ax2.errorbar(
#     cv_stats['k'],
#     cv_stats['avg_error'],
#     yerr=cv_stats['avg_error_std'],
#     linestyle='dotted', marker='s',
#     label='Reduced model, average Euclidean error',
#     color=rgb.tue_violet,
#     capsize=2, markersize=3
# )
ax2.plot(
    cv_stats['k'],
    cv_stats['avg_error'],
    linestyle='-', marker='s',
    # label='Reduced model, average Euclidean error',
    color=rgb.tue_violet,
    markersize=3
)
ax2.fill_between(
    cv_stats['k'],
    cv_stats['scaling_avg_error_lq'],
    cv_stats['scaling_avg_error_uq'],
    color=rgb.tue_violet, alpha=0.2
)
ax2.axhline(
    y=scaling_avg_error,
    color=rgb.tue_violet,
    linestyle='-', alpha=0.5,
    label='Scaling, average Euclidean error'
)
ax2.set_ylabel('Average Euclidean error', color=rgb.tue_violet)
ax2.set_ylim(0, 40)
# xticks = np.arange(0, 45, 5)
# ax2.set_yticks(ticks=xticks, labels=xticks, color=rgb.tue_violet)

# ax.set_xticks(np.arange(2, 22, 2))
ax.set_xlabel(r'$k$')
# ax.legend()

# LOO CV
stats_loo = pd.read_csv('loo_detailed_points.csv', index_col=0).groupby(['left_out_color']).agg(
    avg_error = ('delta_E', 'mean'),
    avg_error_std = ('delta_E', 'std')
).reset_index().merge(
    pd.DataFrame(GT),
    left_on='left_out_color',
    right_on='sample_number',
    how='left'
).assign(
    gt_rgb = lambda df: list(map(
        lambda r, g, b: (r/255, g/255, b/255),
        df['gt__R'], df['gt__G'], df['gt__B']
    ))
)
# print(stats_loo)

ax = axs[1]
ax.bar(
    stats_loo['left_out_color'],
    stats_loo['avg_error'],
    yerr=stats_loo['avg_error_std'], capsize=2,
    color=stats_loo['gt_rgb'],
    linewidth=0.7, edgecolor='black',
)
ax.set_xlim(0.1, 24.9)
ax.set_xlabel('Left-out color index')
ax.set_ylabel('Average Euclidean error ($k=1$)')

plt.savefig('plot_cv_k_out_acc.pdf')

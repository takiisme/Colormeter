import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots.bundles import icml2024
from constants import COLOR

nrows, ncols = 1, 1
plt.rcParams.update(icml2024(column='half', nrows=nrows, ncols=ncols))

n_rows, n_cols = 3, 2
plt.rcParams.update(icml2024(column='half', nrows=n_rows, ncols=n_cols))

df_test = pd.read_csv("Data/test_corrected.csv")

# Plot
def plot_ecdf(ax: plt.Axes, x: np.ndarray, alpha=0.05, **kwargs) -> plt.Axes:
    """
    Compute ECDF for a one-dimensional array of measurements.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes to plot on.
    x : np.ndarray
        One-dimensional array of measurements.
    alpha : float
        Significance level for confidence intervals.
    **kwargs : dict
        Additional keyword arguments for the step plot.
    """
    n = len(x)
    x_sorted = np.sort(x)
    y = np.arange(1, n + 1) / n
    # DKW confidence band
    width = np.sqrt(np.log(2 / alpha) / (2 * n))
    lower = np.maximum(y - width, 0)
    upper = np.minimum(y + width, 1)
    # Plot ECDF
    ax.step(x_sorted, y, where='post', **kwargs)
    # Avoid duplicate label entries
    kwargs.pop('label', None)
    # Plot confidence band
    ax.fill_between(x_sorted, lower, upper, step='post', alpha=0.3, linewidth=0, **kwargs)
    # Plot mean line
    # ax.axvline(np.mean(x), color=kwargs.get('color', 'black'), linestyle='dashed', linewidth=1)
    return ax

fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

eucl_raw = np.sqrt(
    (df_test["color_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["color_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["color_r4_b"] - df_test["gt__b"])**2
)
ax = plot_ecdf(ax, eucl_raw, color=COLOR['raw'], label='Raw')

eucl_scaling = np.sqrt(
    (df_test["scaling_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["scaling_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["scaling_r4_b"] - df_test["gt__b"])**2
)
ax = plot_ecdf(ax, eucl_scaling, color=COLOR['scaling'], label='Scaling')

eucl_full = np.sqrt(
    (df_test["full_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["full_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["full_r4_b"] - df_test["gt__b"])**2
)
ax = plot_ecdf(ax, eucl_full, color=COLOR['full'], label='Full model')

eucl_reduced = np.sqrt(
    (df_test["reduced_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["reduced_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["reduced_r4_b"] - df_test["gt__b"])**2
)
ax = plot_ecdf(ax, eucl_reduced, color=COLOR['reduced'], label='Reduced model')
# TODO: Maybe use different colors for better visibility.

ax.set_xlabel(r'Euclidean error in $L^* a^* b^*$')
ax.set_ylabel('ECDF')
ax.legend()

plt.savefig("Images/plot_error_ecdf.pdf")

# Save some statistics
eucl_raw_mean = np.mean(eucl_raw)
eucl_scaling_mean = np.mean(eucl_scaling)
eucl_full_mean = np.mean(eucl_full)
eucl_reduced_mean = np.mean(eucl_reduced)
eucl_raw_median = np.median(eucl_raw)
eucl_scaling_median = np.median(eucl_scaling)
eucl_full_median = np.median(eucl_full)
eucl_reduced_median = np.median(eucl_reduced)
eucl_raw_max = np.max(eucl_raw)
eucl_scaling_max = np.max(eucl_scaling)
eucl_full_max = np.max(eucl_full)
eucl_reduced_max = np.max(eucl_reduced)
stats = {
    'Raw': {
        'mean': eucl_raw_mean,
        'mean_rel': 0., 
        'median': eucl_raw_median,
        'median_rel': 0.,
        'max': eucl_raw_max,
        'max_rel': 0.,
    },
    'Scaling': {
        'mean': eucl_scaling_mean,
        'mean_rel': (eucl_scaling_mean - eucl_raw_mean) / eucl_raw_mean,
        'median': eucl_scaling_median,
        'median_rel': (eucl_scaling_median - eucl_raw_median) / eucl_raw_median,
        'max': eucl_scaling_max,
        'max_rel': (eucl_scaling_max - eucl_raw_max) / eucl_raw_max,
    },
    'Full model': {
        'mean': eucl_full_mean,
        'mean_rel': (eucl_full_mean - eucl_raw_mean) / eucl_raw_mean,
        'median': eucl_full_median,
        'median_rel': (eucl_full_median - eucl_raw_median) / eucl_raw_median,
        'max': eucl_full_max,
        'max_rel': (eucl_full_max - eucl_raw_max) / eucl_raw_max,
    },
    'Reduced model': {
        'mean': eucl_reduced_mean,
        'mean_rel': (eucl_reduced_mean - eucl_raw_mean) / eucl_raw_mean,
        'median': eucl_reduced_median,
        'median_rel': (eucl_reduced_median - eucl_raw_median) / eucl_raw_median,
        'max': eucl_reduced_max,
        'max_rel': (eucl_reduced_max - eucl_raw_max) / eucl_raw_max,
    },
}
stats = pd.DataFrame(stats).T

stats.round(2).to_csv("error_ecdf_stats.csv")

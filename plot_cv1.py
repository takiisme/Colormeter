import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots.bundles import icml2024
from tueplots.constants.color import rgb
from constants import COLOR


n_rows, n_cols = 1, 2
plt.rcParams.update(icml2024(column='full', nrows=n_rows, ncols=n_cols))

results_k = pd.read_csv('cv_k_out.csv', index_col=0).assign(
    rmses=lambda df: df['mses'] ** 0.5
).groupby('k').agg(
    rmse_mean=('rmses', 'mean'),
    rmse_std=('rmses', 'std')
).reset_index()

# Extract the RMSE for scaling and model on test set
df_test = pd.read_csv('Data/test_corrected.csv')
scaling_error = (
    (df_test['scaling_r4_l'] - df_test['gt__l']) ** 2 +
    (df_test['scaling_r4_a'] - df_test['gt__a']) ** 2 +
    (df_test['scaling_r4_b'] - df_test['gt__b']) ** 2
)
reduced_error = (
    (df_test['reduced_r4_l'] - df_test['gt__l']) ** 2 +
    (df_test['reduced_r4_a'] - df_test['gt__a']) ** 2 +
    (df_test['reduced_r4_b'] - df_test['gt__b']) ** 2
)
scaling_rmse = np.sqrt(np.mean(scaling_error))
reduced_rmse = np.sqrt(np.mean(reduced_error))

fig, axs = plt.subplots(n_rows, n_cols)

# Plot RMSE vs. k
ax = axs[0]
ax.axhline(y=scaling_rmse, color=COLOR['scaling'], linestyle='--', label='scaling')
# ax.axhline(y=reduced_rmse, color=COLOR['reduced'], linestyle='--', label='reduced model')
ax.errorbar(
    results_k['k'],
    results_k['rmse_mean'],
    label='reduced model (CV)',
    yerr=results_k['rmse_std'],
    # fmt='-o',
    color=COLOR['reduced'],
    capsize=2, markersize=4,
    # linewidth=1
)
ax.set_xlabel(r'$k$')
ax.set_ylabel('RMSE')
ax.set_ylim(0, 40)
ax.set_xticks(np.arange(2, 22, 2))
ax.legend()

plt.savefig('plot_cv_k_out.pdf')

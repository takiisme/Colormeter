import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots.bundles import icml2024
# from tueplots.constants.color import rgb
from constants import COLOR

nrows, ncols = 1, 3
plt.rcParams.update(icml2024(column='full', nrows=nrows, ncols=ncols))

df_test = pd.read_csv("Data/test_corrected.csv")

LABEL = {
    'l': r'$L^*$',
    'a': r'$a^*$',
    'b': r'$b^*$',
    'color': 'Raw',
    'scaling': 'Scaling',
    'reduced': 'Reduced model',
}

# =============================================
# Option 1: horizontal KDE plots

fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
for i, channel in enumerate(['l', 'a', 'b']):
    error_raw = df_test[f"color_r4_{channel}"] - df_test[f"gt__{channel}"]
    error_scaling = df_test[f"scaling_r4_{channel}"] - df_test[f"gt__{channel}"]
    error_reduced = df_test[f"reduced_r4_{channel}"] - df_test[f"gt__{channel}"]
    
    sns.kdeplot(
        x=error_raw,
        ax=axs[i],
        color=COLOR['raw'],
        label=LABEL['color'],
        fill=True,
        alpha=0.5,
    )
    sns.kdeplot(
        x=error_scaling,
        ax=axs[i],
        color=COLOR['scaling'],
        label=LABEL['scaling'],
        fill=True,
        alpha=0.5
    )
    sns.kdeplot(
        x=error_reduced,
        ax=axs[i],
        color=COLOR['reduced'],
        label=LABEL['reduced'],
        fill=True,
        alpha=0.5
    )
    axs[i].set_xlabel(f"Error in {LABEL[channel]}")
    axs[i].set_ylabel("")

axs[0].set_ylabel("Density")
# axs[0].legend()
plt.savefig("Images/plot_error_kde.pdf")

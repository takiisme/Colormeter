from util import get_color_col_names, load_data
from correction import CorrectionByScaling, CorrectionByModel
from color_conversion import convert_rgb_cols, convert_to_rgb
from constants import ColorSpace, LightingCondition
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots.bundles import icml2024
from tueplots.constants.color import rgb

n_rows, n_cols = 3, 2
plt.rcParams.update(icml2024(column='half', nrows=n_rows, ncols=n_cols))

df_daylight1 = load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = "daylight"
df = pd.concat([df_daylight1, df_daylight2], ignore_index=True)

corrector_scaling = CorrectionByScaling(space=ColorSpace.RGB)
df = corrector_scaling.apply_correction(df.copy())

color_cols = get_color_col_names(df, r=4, space=ColorSpace.RGB, correction=True, gt=True)

# Store color labels
color_labels = df[['sample_number', 'label']].drop_duplicates().set_index('sample_number')

# Cast into long format for easier plotting
df_long = pd.melt(
    df,
    id_vars=['lighting_condition', 'session_name', 'sample_number', 'capture_index'],
    value_vars=color_cols,
    var_name='variable',
    value_name='value'
)
# Extract channel
df_long['channel'] = df_long['variable'].str[-1]
df_long['type'] = df_long['variable'].str.split('_').str[0]
df_long = df_long.drop(columns=['variable', 'capture_index'])
# Renaming for plot readability
df_long['type'] = df_long['type'].map({
    'color': 'raw',
    'correction': 'corrected',
    'gt': 'true',
})
# Join back color labels
df_long = df_long.join(
    color_labels,
    on='sample_number',
    how='left'
)
channels = ColorSpace.RGB.get_channels()
selected_colors = [
    'Red', 'Green', 'Blue',
    'Yellow', 'Cyan', 'Magenta'
]
df_plot = df_long.query("label in @selected_colors & channel in @channels")

# Here starts the plotting part
fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all')

for ax, color_label in zip(axs.flat, selected_colors):
    df_subset = df_plot.query(f"label == '{color_label}' & lighting_condition == '{LightingCondition.DAYLIGHT.value}'")
    sns.stripplot(
        data=df_subset.query("type != 'true'"),
        x='channel',
        y='value',
        hue='type',
        dodge=True,
        ax=ax,
        palette={
            'raw': rgb.tue_blue,
            'corrected': rgb.tue_red,
        },
        size=2, alpha=0.3
    )
    # sns.violinplot(
    #     data=df_subset.query("type != 'true'"),
    #     x='channel',
    #     y='value',
    #     hue='type',
    #     # split=True,
    #     ax=ax,
    #     palette={
    #         'raw': rgb.tue_blue,
    #         'corrected': rgb.tue_red,
    #     },
    #     inner=None,
    #     linewidth=0,
    # )
    ax.set_xlabel('')
    ax.set_xticks(ticks=[0, 1, 2], labels=[r'$R$', r'$G$', r'$B$'])
    # Add hlines for ground truth
    gt_values = df_subset.query("type == 'true'").set_index('channel')['value'].to_dict()
    for i, (ch, gt) in enumerate(gt_values.items()):
        width = 0.5
        ax.hlines(
            y=gt,
            xmin=i - width,
            xmax=i + width,
            colors='black',
            linestyles='dashed',
        )
    ax.set_title(color_label)
    ax.legend_.remove()

plt.savefig("Images/plot_against_gt.pdf")

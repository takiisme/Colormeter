from data import load_data
from correction import CorrectionByScaling
import pandas as pd
import numpy as np
from constants import ColorSpace, LightingCondition
import seaborn as sns
import matplotlib.pyplot as plt

df_daylight1 = load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = "daylight"
df_dark1 = load_data("Data/Tai4.json")
df_dark1["lighting_condition"] = "dark"
df_dark2 = load_data("Data/Zhi3.json")
df_dark2["lighting_condition"] = "dark"

df = pd.concat([df_daylight1, df_daylight2, df_dark1, df_dark2], ignore_index=True)

corrector = CorrectionByScaling(space=ColorSpace.RGB, r=4)
df_corrected = corrector.predict(df)


# This plot supersedes `plot_error_dist`
def plot_against_gt(
        df: pd.DataFrame,
        space: ColorSpace = ColorSpace.RGB,
        lighting_condition: LightingCondition = LightingCondition.DAYLIGHT,
        r: int = 4
    ) -> sns.FacetGrid:
    """
    Plot raw and corrected measurements against ground truth.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing measurement, correction, and ground truth columns.
    space : ColorSpace
        Color space to plot (RGB, HSV, Lab).
    lighting_condition : LightingCondition
        Lighting condition to filter by (daylight or dark).
    r : int
        Reticle size (0, 2, or 4).

    Returns: sns.FacetGrid
        Seaborn FacetGrid object containing the plots.
    """
    # Store color labels
    color_labels = df[['sample_number', 'label']].drop_duplicates().set_index('sample_number')

    # Cast into long format for easier plotting
    value_vars = df.columns.values.tolist()
    value_vars = [col for col in value_vars if col.startswith(('color', 'correction', 'gt'))]
    df_long = pd.melt(
        df,
        id_vars=['lighting_condition', 'session_name', 'sample_number', 'capture_index'],
        value_vars=value_vars,
        var_name='variable',
        value_name='value'
    )
    # Extract channel
    df_long['channel'] = df_long['variable'].str[-1]
    df_long['type'] = df_long['variable'].str.split('_').str[0]
    df_long['r'] = df_long['variable'].str.extract(r'r(\d+)').astype(int)
    # Keep only the chosen r
    df_long = df_long.query(f"r == {r}")
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

    # Here starts the plotting part
    channels = space.get_channels()
    df_plot = df_long.query("channel in @channels")
    g = sns.FacetGrid(df_plot.query(f"lighting_condition == '{lighting_condition}' & type in ['raw', 'corrected']"), col="label", col_wrap=6)
    # TODO: decide on which plot to use
    # g.map_dataframe(
    #     sns.violinplot,
    #     x="channel",
    #     y="value",
    #     hue="type",
    #     palette={
    #         'raw': 'blue',
    #         'corrected': 'red'
    #     },
    #     alpha=0.5,
    #     split=True,
    #     gap=0.1,
    #     inner=None
    # )
    g.map_dataframe(
        sns.stripplot,
        x="channel",
        y="value",
        hue="type",
        palette={
            'raw': 'blue',
            'corrected': 'red'
        },
        size=4,
        alpha=0.3,
        dodge=True,
    )
    # g.map_dataframe(
    #     sns.boxplot,
    #     x="channel",
    #     y="value",
    #     hue="type",
    #     palette={
    #         'raw': 'blue',
    #         'corrected': 'red'
    #     },
    #     linewidth=1,
    #     fliersize=4,
    #     gap=0.1,
    # )
    # Add hlines for ground truth
    gt_subset = df_plot.query("lighting_condition == 'daylight' & type == 'true'")
    for ax, label in zip(g.axes.flat, g.col_names):
        # break
        gt_values = gt_subset.query(f"label == '{label}'").set_index('channel')['value'].to_dict()
        for i, (ch, gt) in enumerate(gt_values.items()):
            # # Speical case: H of red is ~1, but draw at ~0 -- equivalent, but looks better
            # if label == 'Red' and ch == 'H':
            #     gt = abs(1 - gt)
            width = 0.5
            ax.hlines(
                y=gt,
                xmin=i - width,
                xmax=i + width,
                colors='black',
                linestyles='dashed',
            )
    g.add_legend()
    return g

g = plot_against_gt(df_corrected, space=ColorSpace.RGB, lighting_condition='daylight', r=4)
g.figure.suptitle('daylight, RGB', y=1.02)
plt.savefig('daylight_rgb.png')

g = plot_against_gt(df_corrected, space=ColorSpace.HSV, lighting_condition='daylight', r=4)
g.figure.suptitle('daylight, HSV', y=1.02)
plt.savefig('daylight_hsv.png')

g = plot_against_gt(df_corrected, space=ColorSpace.LAB, lighting_condition='daylight', r=4)
g.figure.suptitle('daylight, Lab', y=1.02)
plt.savefig('daylight_lab.png')

# g = plot_against_gt(df_long, space='rgb', lighting_condition='dark')
# g.figure.suptitle('dark, RGB', y=1.02)
# plt.show()

# g = plot_against_gt(df_long, space='hsv', lighting_condition='dark')
# g.figure.suptitle('dark, HSV', y=1.02)
# plt.show()

# g = plot_against_gt(df_long, space='lab', lighting_condition='dark')
# g.figure.suptitle('dark, Lab', y=1.02)
# plt.show()

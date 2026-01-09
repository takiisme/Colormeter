import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util import get_color_col_names
from color_conversion import convert_rgb_cols
from constants import ColorSpace, LightingCondition


def plot_comparison_grid(df_final_comparison, radius=4, rows=4, cols=6):
    def add_average(df_with_gt_columns: pd.DataFrame, radius: int) -> pd.DataFrame:
        """Calculate average values for the specified radius"""
        # Dynamically create column names based on radius
        avg_cols_to_compute = [
            f'color_r{radius}_R', f'color_r{radius}_G', f'color_r{radius}_B',
            f'correction_r{radius}_R', f'correction_r{radius}_G', f'correction_r{radius}_B'
        ]
        
        df_avg = df_with_gt_columns.groupby('sample_number')[avg_cols_to_compute].mean().reset_index()

        # Rename columns to 'avg_...' to clearly distinguish them
        new_avg_columns_map = {col: 'avg_' + col for col in avg_cols_to_compute}
        df_avg = df_avg.rename(columns=new_avg_columns_map)

        # Merge the df (which now has ground truth) with the averaged color data
        df_final_comparison = pd.merge(df_with_gt_columns, df_avg, on='sample_number', how='left')

        return df_final_comparison

    df_plot = add_average(df_final_comparison, radius)
    
    # Unique samples only
    df_plot = df_plot.drop_duplicates(subset=['sample_number']).reset_index(drop=True)

    # 1. Pre-calculate Hex columns using vectorized utility
    gt_prefix = "gt__"
    uncorr_prefix = f"avg_color_r{radius}_"
    corr_prefix = f"avg_correction_r{radius}_"

    df_plot = convert_rgb_cols(df_plot, prefix=gt_prefix, to=ColorSpace.HEX)
    df_plot = convert_rgb_cols(df_plot, prefix=uncorr_prefix, to=ColorSpace.HEX)
    df_plot = convert_rgb_cols(df_plot, prefix=corr_prefix, to=ColorSpace.HEX)

    num_samples = len(df_plot)
    if num_samples == 0:
        print(f"No data for r{radius}")
        return

    # Change this line:
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5), dpi=80)
    axes = axes.flatten()

    for i, row in df_plot.iterrows():
        if i >= len(axes): break

        # 2. Extract Values
        gt_rgb = row[[f'{gt_prefix}R', f'{gt_prefix}G', f'{gt_prefix}B']].values.astype(float) / 255.0
        uncorr_rgb = row[[f'{uncorr_prefix}R', f'{uncorr_prefix}G', f'{uncorr_prefix}B']].values.astype(float) / 255.0
        corr_rgb = row[[f'{corr_prefix}R', f'{corr_prefix}G', f'{corr_prefix}B']].values.astype(float) / 255.0

        # Grab pre-calculated hex strings
        gt_hex = row[f'{gt_prefix}hex']
        uncorr_hex = row[f'{uncorr_prefix}hex']
        corr_hex = row[f'{corr_prefix}hex']

        # 3. Build Comparison Image
        image = np.zeros((100, 100, 3))
        image[50:100, 0:100] = np.clip(gt_rgb, 0, 1)     # Bottom: GT
        image[0:50, 0:50] = np.clip(uncorr_rgb, 0, 1)    # Top-Left: Uncorrected
        image[0:50, 50:100] = np.clip(corr_rgb, 0, 1)    # Top-Right: Corrected

        ax = axes[i]
        ax.imshow(image)

        # Helper for contrast-based text color
        def text_col(rgb): return 'black' if np.mean(rgb) > 0.5 else 'white'

        # 4. Add Labels using pre-calculated hex
        ax.text(50, 75, gt_hex, color=text_col(gt_rgb), ha='center', fontsize=7, 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))
        ax.text(25, 25, uncorr_hex, color=text_col(uncorr_rgb), ha='center', fontsize=7, 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))
        ax.text(75, 25, corr_hex, color=text_col(corr_rgb), ha='center', fontsize=7, 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

        ax.set_title(f"#{int(row['sample_number'])}: {row['label']}", fontsize=9)
        ax.axis('off')

    # Hide extra axes
    for j in range(num_samples, len(axes)): fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Radius R{radius} Comparison: Ground Truth (Bottom) vs Uncorrected (L) vs Corrected (R)', fontsize=16)
    if radius == 0: plt.savefig('Correction_comparison_R0.png')
    return fig, axes


def plotHSV(df):
    """
    Plots KDE distributions for H, S, and V error in 3 subplots.
    Returns fig, axes.
    """
    def df_HSV(df: pd.DataFrame, radius: int) -> pd.DataFrame:
        gt_prefix = "gt__"
        uncorr_prefix = f"color_r{radius}_"
        corr_prefix = f"correction_r{radius}_"

        df = convert_rgb_cols(df, prefix=gt_prefix, to=ColorSpace.HSV)
        df = convert_rgb_cols(df, prefix=uncorr_prefix, to=ColorSpace.HSV)
        df = convert_rgb_cols(df, prefix=corr_prefix, to=ColorSpace.HSV)

        h_diff_uncorr = np.abs(df[f'{uncorr_prefix}H'] - df[f'{gt_prefix}H'])
        df['H_error_uncorr'] = np.minimum(h_diff_uncorr, 1 - h_diff_uncorr)

        h_diff_corr = np.abs(df[f'{corr_prefix}H'] - df[f'{gt_prefix}H'])
        df['H_error_corr'] = np.minimum(h_diff_corr, 1 - h_diff_corr)

        for comp in ['S', 'V']:
            df[f'{comp}_error_uncorr'] = df[f'{uncorr_prefix}{comp}'] - df[f'{gt_prefix}{comp}']
            df[f'{comp}_error_corr'] = df[f'{corr_prefix}{comp}'] - df[f'{gt_prefix}{comp}']

        return df

    # Prepare data
    df_extended = df_HSV(df, radius=4)
    df_extended = df_HSV(df_extended, radius=2)
    df = df_HSV(df_extended, radius=0)

    hsv_components = ['H', 'S', 'V']

    # === 3 SUBPLOTS ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, component in zip(axes, hsv_components):
        uncorr_col = f'{component}_error_uncorr'
        corr_col = f'{component}_error_corr'

        sns.kdeplot(x=df[uncorr_col], fill=True, alpha=0.2,
                    label='Uncorrected', ax=ax)
        sns.kdeplot(x=df[corr_col], fill=True, alpha=0.2,
                    label='Corrected', ax=ax)

        for col, label_prefix, y_offset in [
            (uncorr_col, 'Uncorr', 0.9),
            (corr_col, 'Corr', 0.75)
        ]:
            m = df[col].mean()
            s = df[col].std()
            ax.axvline(m, linestyle='dashed', linewidth=1)
            ax.text(m + 0.01, ax.get_ylim()[1] * y_offset,
                    f'Mean {label_prefix}: {m:.3f}')
            ax.text(m + 0.01, ax.get_ylim()[1] * (y_offset - 0.05),
                    f'Std {label_prefix}: {s:.3f}')

        ax.set_title(f'{component} Error')
        ax.set_xlabel(f'Error ({component})')
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[0].set_ylabel('Density')
    axes[0].legend()

    fig.suptitle('Distribution of HSV Error: Uncorrected vs. Corrected', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, axes


def plotHSVError(df):
    """
    Visualizes error vs angles in a single 3x2 subplot figure.
    Returns fig, axes.
    """
    plt.close('all')

    df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.HSV)
    df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.LAB)
    df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.HEX)

    # Detect corrected prefix
    possible_prefixes = ['correction_r4_', 'correction_r2_', 'correction_r0_']
    corr_prefix = next((p for p in possible_prefixes if f'{p}R' in df.columns), None)

    if corr_prefix is None:
        raise KeyError("No correction RGB columns found.")

    df = convert_rgb_cols(df, prefix=corr_prefix, to=ColorSpace.HSV)

    # Compute errors
    channels = ['H', 'S', 'V']
    for channel in channels:
        df[f'{channel}_error_corr'] = df[f'{corr_prefix}{channel}'] - df[f'gt__{channel}']

        print(f"\n--- {channel} Channel Analysis (using {corr_prefix}) ---")
        for angle in ['pitch', 'roll']:
            corr = df[f'{channel}_error_corr'].corr(df[angle])
            print(f"Pearson correlation with {angle.capitalize()}: {corr:.4f}")

    # === 3x2 SUBPLOTS ===
    fig, axes = plt.subplots(3, 2, figsize=(15, 25), sharex='col')

    hex_col = 'gt__hex'
    label_color_map = (
        df[['label', hex_col]]
        .drop_duplicates()
        .set_index('label')[hex_col]
        .to_dict()
    )

    for row, channel in enumerate(channels):
        error_col = f'{channel}_error_corr'

        for col, angle in enumerate(['pitch', 'roll']):
            ax = axes[row, col]

            sns.scatterplot(
                x=angle, y=error_col, hue='label', data=df,
                ax=ax, alpha=0.6, s=50, palette=label_color_map,
                legend='full' if (row == 0 and col == 0) else False
            )

            sns.regplot(
                x=angle, y=error_col, data=df, scatter=False,
                color='red', line_kws={'linestyle': '--', 'alpha': 0.7, 'lw': 2},
                ax=ax
            )

            ax.set_title(f'{channel} Error vs. {angle.capitalize()}')
            ax.set_xlabel(f'{angle.capitalize()} Angle (degrees)')
            ax.set_ylabel(f'Error ({channel})')
            ax.grid(True, linestyle='--', alpha=0.7)

    # Legend only once
    if axes[0, 0].get_legend():
        axes[0, 0].legend(title='Sample Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle('Corrected HSV Error vs. Camera Angles', fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    return fig, axes


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
    try:
        color_cols = get_color_col_names(df, r=r, space=space, correction=True, gt=True)
    except AssertionError:
        # If columns do not exist, add them
        df = convert_rgb_cols(df, prefix=f'color_r{r}_', to=space)
        df = convert_rgb_cols(df, prefix=f'correction_r{r}_', to=space)
        df = convert_rgb_cols(df, prefix='gt__', to=space)
        color_cols = get_color_col_names(df, r=r, space=space, correction=True, gt=True)
    
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

    # Here starts the plotting part
    channels = space.get_channels()
    df_plot = df_long.query("channel in @channels")
    g = sns.FacetGrid(df_plot.query(f"lighting_condition == '{lighting_condition.value}' & type in ['raw', 'corrected']"), col="label", col_wrap=6)
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

# ECDF plot helper functions
def ecdf(x, alpha=0.05):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(x)
    x_sorted = np.sort(x)
    y = np.arange(1, n + 1) / n
    # DKW confidence intervals
    width = np.sqrt(np.log(2 / alpha) / (2 * n))
    lower = np.maximum(y - width, 0)
    upper = np.minimum(y + width, 1)
    return x_sorted, y, lower, upper

def plot_ecdf(eucl_raw, eucl_corrected, alpha=0.05, ax=None):
    """Compute ECDF for a one-dimensional array of measurements."""
    if ax is None:
        ax = plt.gca()
    x_sorted, y, lower, upper = ecdf(eucl_raw, alpha=alpha)
    ax.step(x_sorted, y, label='raw', where='post', color='blue')
    ax.fill_between(x_sorted, lower, upper, step='post', alpha=0.2, color='blue')
    x_sorted, y, lower, upper = ecdf(eucl_corrected, alpha=alpha)
    ax.step(x_sorted, y, label='corrected', where='post', color='red')
    ax.fill_between(x_sorted, lower, upper, step='post', alpha=0.2, color='red')
    return ax

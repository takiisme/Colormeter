import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from color_conversion import convert_rgb_cols
from constants import ColorSpace, LightingCondition
from data import assert_cols_exist


def plot_comparison_grid(
        df: pd.DataFrame,
        r: int,
        rows: int = 4, cols: int = 6
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots measured, correction, and ground truth colors, averaged over the sample, as small color patches.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing measurement, correction, and ground truth columns.
    r : int
        Reticle size (0, 2, or 4).
    rows : int
        Number of rows in the grid.
        `rows` x `cols` = 24 is recommended.
    cols : int
        Number of columns in the grid.
    """

    # Helper function to add average color columns
    def add_averge(df_with_gt_columns: pd.DataFrame) -> pd.DataFrame:
        avg_cols_to_compute = [
            'color_r0_R', 'color_r0_G', 'color_r0_B',
            'correction_r0_R', 'correction_r0_G', 'correction_r0_B',
            'color_r2_R', 'color_r2_G', 'color_r2_B',
            'correction_r2_R', 'correction_r2_G', 'correction_r2_B',
            'color_r4_R', 'color_r4_G', 'color_r4_B',
            'correction_r4_R', 'correction_r4_G', 'correction_r4_B'
        ]
        df_avg = df_with_gt_columns.groupby('sample_number')[avg_cols_to_compute].mean().reset_index()

        # Rename columns to 'avg_...' to clearly distinguish them
        new_avg_columns_map = {col: 'avg_' + col for col in avg_cols_to_compute}
        df_avg = df_avg.rename(columns=new_avg_columns_map)

        # Merge the df (which now has ground truth) with the averaged color data
        df = pd.merge(df_with_gt_columns, df_avg, on='sample_number', how='left')

        return df

    df_plot = add_averge(df)
    
    # Unique samples only
    df_plot = df_plot.drop_duplicates(subset=['sample_number']).reset_index(drop=True)

    # 1. Pre-calculate Hex columns using vectorized utility
    # Using the exact prefixes from your naming convention
    gt_prefix = "gt__"
    uncorr_prefix = f"avg_color_r{r}_"
    corr_prefix = f"avg_correction_r{r}_"

    df_plot = convert_rgb_cols(df_plot, prefix=gt_prefix, to=ColorSpace.HEX)
    df_plot = convert_rgb_cols(df_plot, prefix=uncorr_prefix, to=ColorSpace.HEX)
    df_plot = convert_rgb_cols(df_plot, prefix=corr_prefix, to=ColorSpace.HEX)

    num_samples = len(df_plot)
    if num_samples == 0:
        print(f"No data for r{r}")
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
        # Positioned in the center of each respective quadrant
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
    plt.suptitle(f'Comparison: Ground Truth (Bottom) vs Uncorrected (L) vs Corrected (R), r={r}', fontsize=16)
    return fig, axes


def HSV_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized HSV error analysis.
    Uses convert_rgb_cols and handles dynamic correction prefixes.
    """
    # 1. Process Ground Truth (gt__)
    df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.HSV)
    df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.HEX)
    
    # 2. Process Corrected Colors: auto-detect prefix
    possible_prefixes = ['correction_r4_', 'correction_r2_', 'correction_r0_']
    corr_prefix = next((p for p in possible_prefixes if f'{p}R' in df.columns), None)

    if corr_prefix is None:
        raise KeyError("No correction RGB columns found. Ensure corrections are applied first.")

    # Convert corrected RGB to HSV
    df = convert_rgb_cols(df, prefix=corr_prefix, to=ColorSpace.HSV)

    # 3. Calculate Errors and Run Correlations
    channels = ['H', 'S', 'V']
    
    for channel in channels:
        error_col = f'{channel}_error_corr'
        gt_col = f'gt__{channel}'
        corr_col = f'{corr_prefix}{channel}'
        
        # Vectorized subtraction
        df[error_col] = df[corr_col] - df[gt_col]
        
        # Stats Output
        print(f"\n--- {channel} Channel Analysis (using {corr_prefix}) ---")
        for angle in ['pitch', 'roll']:
            correlation = df[error_col].corr(df[angle])
            print(f"Pearson correlation with {angle.capitalize()}: {correlation:.4f}")
        
        # Plotting
        plotHSVError(df, option=channel)

    return df


def plotHSVError(df: pd.DataFrame, option: str = 'V') -> tuple[plt.Figure, plt.Axes]:
    """
    Visualizes error vs angles with a fresh figure for each call.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing error and angle columns.
    option : str
        Color channel option ('H', 'S', or 'V').
    """
    plt.close('all') # Clear previous figures to prevent memory/render issues
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    error_col = f'{option}_error_corr'
    hex_col = 'gt__hex'

    # Map labels to hex colors for the plot palette
    label_color_map = df[['label', hex_col]].drop_duplicates().set_index('label')[hex_col].to_dict()

    for i, angle in enumerate(['pitch', 'roll']):
        # Scatter points colored by ground truth
        sns.scatterplot(
            x=angle, y=error_col, hue='label', data=df, 
            ax=axes[i], alpha=0.6, s=50, palette=label_color_map, 
            legend='full' if i == 0 else False
        )
        
        # Trend line
        sns.regplot(
            x=angle, y=error_col, data=df, scatter=False, 
            color='red', line_kws={'linestyle':'--', 'alpha':0.7, 'lw':2}, ax=axes[i]
        )
        
        axes[i].set_title(f'Corrected {option} Error vs. {angle.capitalize()}')
        axes[i].set_xlabel(f'{angle.capitalize()} Angle (degrees)')
        axes[i].set_ylabel(f'Error ({option})')
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Place legend only on the first plot
    if axes[0].get_legend():
        axes[0].legend(title='Sample Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.95, 1])
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
        assert_cols_exist(df, r=r, space=space, correction=True, gt=True)
    except AssertionError:
        # If columns do not exist, add them
        df = convert_rgb_cols(df, prefix=f'color_r{r}_', to=space)
        df = convert_rgb_cols(df, prefix=f'correction_r{r}_', to=space)
        df = convert_rgb_cols(df, prefix='gt__', to=space)

    
    # Store color labels
    color_labels = df[['sample_number', 'label']].drop_duplicates().set_index('sample_number')

    # Cast into long format for easier plotting
    channels = space.get_channels()
    value_vars = [
        f'color_r{r}_{ch}' for ch in channels
    ] + [
        f'correction_r{r}_{ch}' for ch in channels
    ] + [
        f'gt__{ch}' for ch in channels
    ]
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

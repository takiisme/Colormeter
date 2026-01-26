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
    fig, axes = plt.subplots(rows, cols)
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

# TODO: make r option work
def plotHSV(df, r=4):
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
    # df_extended = df_HSV(df, radius=4)
    # df_extended = df_HSV(df_extended, radius=2)
    df = df_HSV(df, radius=r)

    hsv_components = ['H', 'S', 'V']

    # === 3 SUBPLOTS ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = {'Uncorrected': 'blue', 'Corrected': 'red'}

    for ax, component in zip(axes, hsv_components):
        uncorr_col = f'{component}_error_uncorr'
        corr_col = f'{component}_error_corr'

        sns.kdeplot(x=df[uncorr_col], fill=True, alpha=0.2,
                    label='Raw', ax=ax, color=colors['Uncorrected'])
        sns.kdeplot(x=df[corr_col], fill=True, alpha=0.2,
                    label='Corrected', ax=ax, color=colors['Corrected'])

        for col, label, y_offset in [
            (uncorr_col, 'Uncorrected', 0.9),
            (corr_col, 'Corrected', 0.75)
        ]:
            m = df[col].mean()
            s = df[col].std()
            ax.axvline(m, linestyle='dashed', linewidth=1, color=colors[label])
            ax.text(m + 0.01, ax.get_ylim()[1] * y_offset,
                    f'Mean: {m:.3f}', color=colors[label])
            ax.text(m + 0.01, ax.get_ylim()[1] * (y_offset - 0.05),
                    f'Std: {s:.3f}', color=colors[label])

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


def plot_targeted_results(targeted_results, save_path=None):
    """Plot the results of targeted cross-validation."""
    valid_results = {k: v for k, v in targeted_results.items() 
                    if not np.isnan(v['mse']) and 'error' not in v}
    
    if not valid_results:
        print("\nNo valid results to plot.")
        return
    
    # Prepare data for plotting
    names = list(valid_results.keys())
    mses = [valid_results[name]['mse'] for name in names]
    n_tests = [valid_results[name]['n_test'] for name in names]
    n_trains = [valid_results[name]['n_train'] for name in names]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot for MSE
    bars1 = ax1.bar(range(len(names)), mses, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Test Set')
    ax1.set_ylabel('Average MSE')
    ax1.set_title('Targeted Cross-Validation: Average MSE')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mse in zip(bars1, mses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mse:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Bar plot for sample counts
    x = np.arange(len(names))
    width = 0.35
    bars2 = ax2.bar(x - width/2, n_trains, width, label='Training Samples', color='lightgreen', alpha=0.8)
    bars3 = ax2.bar(x + width/2, n_tests, width, label='Test Samples', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Test Set')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Counts in Targeted Cross-Validation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close()  # Close the figure
    else:
        plt.show()

def plot_k_out_results(results_by_k, save_path=None):
    """Plot the results of k-color-out cross-validation."""
    valid_k_values = [k for k, v in results_by_k.items() if v is not None]
    
    if not valid_k_values:
        print("No valid results to plot.")
        return
    
    mean_mses = [results_by_k[k]['mean_mse'] for k in valid_k_values]
    std_mses = [results_by_k[k]['std_mse'] for k in valid_k_values]
    min_mses = [results_by_k[k]['min_mse'] for k in valid_k_values]
    max_mses = [results_by_k[k]['max_mse'] for k in valid_k_values]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean MSE with error bars
    ax1.errorbar(valid_k_values, mean_mses, yerr=std_mses, fmt='-o', 
                 capsize=5, capthick=2, ecolor='red', color='blue', linewidth=2)
    ax1.set_xlabel('Number of Test Samples (k)')
    ax1.set_ylabel('Average MSE')
    ax1.set_title('Average MSE vs. Number of Test Samples')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Min, Mean, Max MSE
    ax2.fill_between(valid_k_values, min_mses, max_mses, alpha=0.3, color='gray', label='Range')
    ax2.plot(valid_k_values, mean_mses, 'b-o', label='Mean')
    ax2.set_xlabel('Number of Test Samples (k)')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE Range vs. Number of Test Samples')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of MSE distributions
    mse_data = [results_by_k[k]['mses'] for k in valid_k_values]
    ax3.boxplot(mse_data, positions=valid_k_values, widths=0.6)
    ax3.set_xlabel('Number of Test Samples (k)')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE Distribution by k')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Standard Deviation
    ax4.bar(valid_k_values, std_mses, alpha=0.7, color='orange')
    ax4.set_xlabel('Number of Test Samples (k)')
    ax4.set_ylabel('Standard Deviation of MSE')
    ax4.set_title('MSE Variability by k')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close()  # Close the figure
    else:
        plt.show()


def plot_leave_one_out_results(results_df):
    """Plot leave-one-color-out analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: MSE for each color
    colors = results_df['left_out_color'].astype(str)
    mses = results_df['mse']
    
    bars = ax1.bar(colors, mses, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Left-Out Color')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE When Leaving Out Each Color')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Color the top 3 highest MSE bars
    top_n = min(3, len(results_df))
    for i in range(top_n):
        bars[i].set_color('red')
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., 
                bars[i].get_height() + 0.01,
                f'#{i+1}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MSE vs GT RGB values
    ax2.scatter(results_df['gt_R_mean'], results_df['mse'], 
                c='red', alpha=0.6, label='R', s=100)
    ax2.scatter(results_df['gt_G_mean'], results_df['mse'], 
                c='green', alpha=0.6, label='G', s=100)
    ax2.scatter(results_df['gt_B_mean'], results_df['mse'], 
                c='blue', alpha=0.6, label='B', s=100)
    
    ax2.set_xlabel('Ground Truth RGB Value')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE vs Ground Truth RGB Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("leave_one_out_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved as 'leave_one_out_analysis.png'")
    
    # Create a summary of problematic colors
    problematic = results_df.head(3)
    print("\n" + "="*80)
    print("TOP 3 PROBLEMATIC COLORS (Highest MSE when left out):")
    print("="*80)
    for i, (_, row) in enumerate(problematic.iterrows(), 1):
        print(f"\n{i}. Color {row['left_out_color']}:")
        print(f"   MSE: {row['mse']:.4f}")
        print(f"   GT RGB: ({row['gt_R_mean']:.0f}, {row['gt_G_mean']:.0f}, {row['gt_B_mean']:.0f})")
        print(f"   Measured RGB: ({row['color_r4_R_mean']:.0f}, {row['color_r4_G_mean']:.0f}, {row['color_r4_B_mean']:.0f})")

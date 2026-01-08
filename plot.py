import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import data
import color_conversion
from constants import ColorSpace


def plot_comparison_grid(df_final_comparison, radius, rows=4, cols=6):
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
        df_final_comparison = pd.merge(df_with_gt_columns, df_avg, on='sample_number', how='left')

        return df_final_comparison

    df_plot = add_averge(df_final_comparison)
    
    # Unique samples only
    df_plot = df_plot.drop_duplicates(subset=['sample_number']).reset_index(drop=True)

    # 1. Pre-calculate Hex columns using vectorized utility
    # Using the exact prefixes from your naming convention
    gt_prefix = "gt__"
    uncorr_prefix = f"avg_color_r{radius}_"
    corr_prefix = f"avg_correction_r{radius}_"

    df_plot = color_conversion.convert_rgb_cols(df_plot, prefix=gt_prefix, to=ColorSpace.HEX)
    df_plot = color_conversion.convert_rgb_cols(df_plot, prefix=uncorr_prefix, to=ColorSpace.HEX)
    df_plot = color_conversion.convert_rgb_cols(df_plot, prefix=corr_prefix, to=ColorSpace.HEX)

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

        df = color_conversion.convert_rgb_cols(df, prefix=gt_prefix, to=ColorSpace.HSV)
        df = color_conversion.convert_rgb_cols(df, prefix=uncorr_prefix, to=ColorSpace.HSV)
        df = color_conversion.convert_rgb_cols(df, prefix=corr_prefix, to=ColorSpace.HSV)

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

    df = color_conversion.convert_rgb_cols(df, prefix="gt__", to=ColorSpace.HSV)
    df = color_conversion.convert_rgb_cols(df, prefix="gt__", to=ColorSpace.LAB)
    df = color_conversion.convert_rgb_cols(df, prefix="gt__", to=ColorSpace.HEX)

    # Detect corrected prefix
    possible_prefixes = ['correction_r4_', 'correction_r2_', 'correction_r0_']
    corr_prefix = next((p for p in possible_prefixes if f'{p}R' in df.columns), None)

    if corr_prefix is None:
        raise KeyError("No correction RGB columns found.")

    df = color_conversion.convert_rgb_cols(df, prefix=corr_prefix, to=ColorSpace.HSV)

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

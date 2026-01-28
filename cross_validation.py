import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Any, Optional
from correction import CorrectionByModel, CorrectionByScaling
from color_conversion import convert_rgb_cols, convert_to_rgb
from constants import ColorSpace, LightingCondition
from plot import plot_k_out_results, plot_targeted_results, plot_leave_one_out_results
from util import load_data
from tqdm import tqdm

# TODO: Consider tracking probability of low Delta E error instead of MSE

# def cross_validate_model_k_out(model_class, df_train, k_min=1, k_max=20, iterations_per_k=5, **model_kwargs):
#     """
#     Perform k-color-out cross-validation as specified.
    
#     For each k from k_min to k_max, run iterations_per_k iterations.
#     For each iteration, leave k random samples out for testing.
    
#     Parameters
#     ----------
#     model_class : class
#         The model class (CorrectionByModel)
#     df_train : pd.DataFrame
#         Training dataframe with measurements
#     k_min : int
#         Minimum k value (default: 1)
#     k_max : int
#         Maximum k value (default: 20)
#     iterations_per_k : int
#         Number of iterations to run for each k value
#     **model_kwargs : dict
#         Keyword arguments for initializing the model
        
#     Returns
#     -------
#     dict : Dictionary with CV results by k value
#     """
#     unique_sample_numbers = df_train['sample_number'].unique()
#     print(f"Unique sample numbers: {unique_sample_numbers}")
#     print(f"Number of unique sample numbers: {len(unique_sample_numbers)}")
    
#     all_sample_numbers = unique_sample_numbers.tolist()
#     total_samples = len(all_sample_numbers)
    
#     # Check k_max doesn't exceed available samples
#     if k_max > total_samples:
#         print(f"Warning: k_max ({k_max}) exceeds total samples ({total_samples}). Reducing k_max to {total_samples}.")
#         k_max = total_samples
    
#     # Initialize results storage
#     results_by_k = []
    
#     for k in range(k_min, k_max + 1):
#         print(f"\n{'='*60}")
#         print(f"Running k-Color-Out Cross-Validation for k = {k}")
#         print(f"{'='*60}")
        
#         iteration_mses = []
#         iteration_results = []
        
#         for iteration in range(iterations_per_k):
#             print(f"\n  Iteration {iteration + 1}/{iterations_per_k}")
            
#             # a. Randomly select k unique sample numbers for testing (without replacement)
#             test_samples = random.sample(all_sample_numbers, k)
#             print(f"  Test samples: {sorted(test_samples)}")
            
#             # b. Remaining samples for training
#             train_samples = [s for s in all_sample_numbers if s not in test_samples]
            
#             # c. Create train and test dataframes
#             cv_df_train = df_train[df_train['sample_number'].isin(train_samples)].copy()
#             cv_df_test = df_train[df_train['sample_number'].isin(test_samples)].copy()
            
#             # Check if we have enough training data
#             if len(cv_df_train) < 10:
#                 print(f"  Warning: Training set too small ({len(cv_df_train)} rows). Skipping iteration.")
#                 continue
            
#             # Initialize and train model
#             model = model_class(**model_kwargs)
#             model.train(cv_df_train)
            
#             # Apply correction to test set
#             cv_df_test_corrected = model.apply_correction(cv_df_test.copy())
            
#             # Calculate MSE based on color space
#             mse = calculate_mse_for_model(model, cv_df_test_corrected)
            
#             print(f"  Average MSE: {mse:.4f}")
            
#             iteration_mses.append(mse)
#             iteration_results.append({
#                 'iteration': iteration + 1,
#                 'test_samples': test_samples,
#                 'mse': mse,
#                 'n_train': len(cv_df_train),
#                 'n_test': len(cv_df_test)
#             })

#             results_by_k.append({
#                 'k': k,
#                 'iteration': iteration + 1,
#                 'mses': mse
#             })
        
#         # # Store results for this k value
#         # if iteration_mses:
#         #     results_by_k[k] = {
#         #         'mses': iteration_mses,
#         #         'mean_mse': ,
#         #         'std_mse': np.std(iteration_mses),
#         #         'median_mse': np.median(iteration_mses),
#         #         'min_mse': np.min(iteration_mses),
#         #         'max_mse': np.max(iteration_mses),
#         #         'iterations_completed': len(iteration_mses),
#         #         'iteration_details': iteration_results
#         #     }
            
#         #     print(f"\n  Summary for k={k}:")
#         #     print(f"    Mean MSE: {results_by_k[k]['mean_mse']:.4f}")
#         #     print(f"    Std MSE: {results_by_k[k]['std_mse']:.4f}")
#         #     print(f"    Min MSE: {results_by_k[k]['min_mse']:.4f}")
#         #     print(f"    Max MSE: {results_by_k[k]['max_mse']:.4f}")
#         #     print(f"    Iterations completed: {results_by_k[k]['iterations_completed']}/{iterations_per_k}")
#         # else:
#         #     results_by_k[k] = None
#         #     print(f"\n  No valid iterations completed for k={k}")
    
#     # Plot overall results
#     plot_k_out_results(results_by_k)
#     pd.DataFrame(results_by_k).to_csv('cv_k_out.csv')
    
#     # # Print final summary
#     # print_k_out_summary(results_by_k)
    
#     return results_by_k

# def cross_validate_model_k_out_fix(model_class, df_train, k_min=1, k_max=20, iterations_per_k=5, **model_kwargs):
#     unique_sample_numbers = df_train['sample_number'].unique()
#     all_sample_numbers = unique_sample_numbers.tolist()
#     total_samples = len(all_sample_numbers)
    
#     # Check k_max doesn't exceed available samples
#     if k_max > total_samples: 
#         k_max = total_samples
    
#     # Initialize results storage
#     rgb_cols = ['color_r4_R', 'color_r4_G', 'color_r4_B']
#     lab_ground_truth_cols = ['gt__l', 'gt__a', 'gt__b']
#     lab_corrected = ['correction_r4_l', 'correction_r4_a', 'correction_r4_b']

#     result_by_k = []
#     detailed_logs = []

#     total_iterations = (k_max - k_min + 1) * iterations_per_k
#     pbar = tqdm(total=total_iterations, desc="K-Color-Out CV Progress")
    
#     for k in range(k_min, k_max + 1):
#         pbar.set_description(f"CV (k={k})")
#         for iteration in range(iterations_per_k):    
#             pbar.update(1)        
#             # a. Randomly select k unique sample numbers for testing (without replacement)
#             test_samples = random.sample(all_sample_numbers, k)
            
#             # b. Remaining samples for training
#             train_samples = [s for s in all_sample_numbers if s not in test_samples]
            
#             # c. Create train and test dataframes
#             cv_df_train = df_train[df_train['sample_number'].isin(train_samples)].copy()
#             cv_df_test = df_train[df_train['sample_number'].isin(test_samples)].copy()
            
#             # Check if we have enough training data
#             if len(cv_df_train) < 5:
#                 print(f"  Warning: Training set too small ({len(cv_df_train)} rows). Skipping iteration.")
#                 continue
            
#             # Initialize and train model
#             model = model_class(**model_kwargs)
#             model.train(cv_df_train)
            
#             # Apply correction to test set
#             cv_df_test_corrected = model.apply_correction(cv_df_test.copy())
#             #print(cv_df_test_corrected.columns)
            
#             # Calculate Euclidean distance (delta E) in LAB space
#             diffs = cv_df_test_corrected[lab_corrected].values - cv_df_test_corrected[lab_ground_truth_cols].values
#             distances = np.linalg.norm(diffs, axis=1)
#             accuracy = np.mean(distances < 2.0)  # Percentage of samples with delta E < 2
            
#             for i, (_, row) in enumerate(cv_df_test_corrected.iterrows()):
#                 detailed_logs.append({
#                     'k': k,
#                     'iteration': iteration + 1,
#                     'sample_id': row['sample_number'],
#                     'R_measured': row['color_r2_R'],
#                     'G_measured': row['color_r2_G'],
#                     'B_measured': row['color_r2_B'],
#                     'L_corrected': row['correction_r4_l'],
#                     'a_corrected': row['correction_r4_a'],
#                     'b_corrected': row['correction_r4_b'],
#                     'L_ground_truth': row['gt__l'],
#                     'a_ground_truth': row['gt__a'],
#                     'b_ground_truth': row['gt__b'],
#                     'delta_E': distances[i],
#                     'is_accurate': distances[i] < 2.0
#                 })
#             result_by_k.append({'k': k, 'iteration': iteration + 1, 'accuracy': accuracy})

#     pbar.close()
#     pd.DataFrame(detailed_logs).to_csv('cv_k_out_detailed.csv', index=False)
    
#     return result_by_k

def cross_validate_model_k_out(model_class, df_train, k_min=1, k_max=20, iterations_per_k=5, **model_kwargs):
    unique_sample_numbers = df_train['sample_number'].unique()
    all_sample_numbers = unique_sample_numbers.tolist()
    total_samples = len(all_sample_numbers)
    
    # Check k_max doesn't exceed available samples
    if k_max > total_samples: 
        k_max = total_samples
    
    # Initialize results storage
    rgb_cols = ['color_r4_R', 'color_r4_G', 'color_r4_B']
    lab_ground_truth_cols = ['gt__l', 'gt__a', 'gt__b']
    lab_corrected = ['correction_r4_l', 'correction_r4_a', 'correction_r4_b']

    detailed_logs = []
    # This list stores the accuracy/mse per iteration to build the plotter dict later
    iteration_stats = [] 

    total_iterations = (k_max - k_min + 1) * iterations_per_k
    pbar = tqdm(total=total_iterations, desc="K-Color-Out CV Progress")
    
    for k in range(k_min, k_max + 1):
        pbar.set_description(f"CV (k={k})")
        for iteration in range(iterations_per_k):    
            pbar.update(1)        
            # a. Randomly select k unique sample numbers for testing (without replacement)
            test_samples = random.sample(all_sample_numbers, k)
            
            # b. Remaining samples for training
            train_samples = [s for s in all_sample_numbers if s not in test_samples]
            
            # c. Create train and test dataframes
            cv_df_train = df_train[df_train['sample_number'].isin(train_samples)].copy()
            cv_df_test = df_train[df_train['sample_number'].isin(test_samples)].copy()
            
            # Check if we have enough training data
            if len(cv_df_train) < 5:
                print(f"  Warning: Training set too small ({len(cv_df_train)} rows). Skipping iteration.")
                continue
            
            # Initialize and train model
            model = model_class(**model_kwargs)
            model.train(cv_df_train)
            
            # Apply correction to test set
            cv_df_test_corrected = model.apply_correction(cv_df_test.copy())
            
            # Calculate Euclidean distance (delta E) in LAB space
            diffs = cv_df_test_corrected[lab_corrected].values - cv_df_test_corrected[lab_ground_truth_cols].values
            distances = np.linalg.norm(diffs, axis=1)
            
            # Calculate metrics for this iteration
            accuracy = np.mean(distances < 2.0)
            mse = np.mean(distances**2)
            
            for i, (_, row) in enumerate(cv_df_test_corrected.iterrows()):
                detailed_logs.append({
                    'k': k,
                    'iteration': iteration + 1,
                    'sample_id': row['sample_number'],
                    'R_measured': row['color_r2_R'], 
                    'G_measured': row['color_r2_G'],
                    'B_measured': row['color_r2_B'],
                    'L_corrected': row['correction_r4_l'],
                    'a_corrected': row['correction_r4_a'],
                    'b_corrected': row['correction_r4_b'],
                    'L_ground_truth': row['gt__l'],
                    'a_ground_truth': row['gt__a'],
                    'b_ground_truth': row['gt__b'],
                    'delta_E': distances[i],
                    'is_accurate': distances[i] < 2.0
                })
            
            iteration_stats.append({'k': k, 'mse': mse, 'accuracy': accuracy})

    pbar.close()
    
    # Save detailed logs
    df_detailed = pd.DataFrame(detailed_logs)
    df_detailed.to_csv('cv_k_out_detailed.csv', index=False)
    
    # --- TRANSFORM TO PLOTTER DICTIONARY FORMAT ---
    # This rebuilds the dictionary your plotting function requires
    results_by_k = {}
    stats_df = pd.DataFrame(iteration_stats)
    
    for k in range(k_min, k_max + 1):
        k_subset = stats_df[stats_df['k'] == k]
        if not k_subset.empty:
            results_by_k[k] = {
                'mean_mse': k_subset['mse'].mean(),
                'std_mse': k_subset['mse'].std() if len(k_subset) > 1 else 0,
                'min_mse': k_subset['mse'].min(),
                'max_mse': k_subset['mse'].max(),
                'mses': k_subset['mse'].tolist(),
                'mean_accuracy': k_subset['accuracy'].mean()
            }
        else:
            results_by_k[k] = None

    # Save image instead of showing
    plot_k_out_results(results_by_k, save_path="k_out_analysis.png")
    
    return results_by_k

def calculate_mse_for_model(model, df_corrected):
    """
    Calculate MSE for corrected dataframe based on model's color space.
    
    Parameters
    ----------
    model : CorrectionByModel
        Trained model
    df_corrected : pd.DataFrame
        DataFrame with corrected values
        
    Returns
    -------
    float : Average MSE across channels
    """
    corr_prefix = f'correction_r{model.r}_'
    
    if model.space.name == 'RGB':
        mse_r = mean_squared_error(df_corrected['gt__R'], 
                                  df_corrected[f'{corr_prefix}R'])
        mse_g = mean_squared_error(df_corrected['gt__G'], 
                                  df_corrected[f'{corr_prefix}G'])
        mse_b = mean_squared_error(df_corrected['gt__B'], 
                                  df_corrected[f'{corr_prefix}B'])
        return (mse_r + mse_g + mse_b) #/ 3.0
        
    elif model.space.name == 'LAB':
        mse_l = mean_squared_error(df_corrected['gt__l'], 
                                  df_corrected[f'{corr_prefix}l'])
        mse_a = mean_squared_error(df_corrected['gt__a'], 
                                  df_corrected[f'{corr_prefix}a'])
        mse_b = mean_squared_error(df_corrected['gt__b'], 
                                  df_corrected[f'{corr_prefix}b'])
        return (mse_l + mse_a + mse_b) #/ 3.0
    
    return np.nan


def print_k_out_summary(results_by_k):
    """Print summary of k-out cross-validation results."""
    print("\n" + "="*80)
    print("K-COLOR-OUT CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(f"{'k':>4} {'Mean MSE':>10} {'Std MSE':>10} {'Min MSE':>10} {'Max MSE':>10} {'Iterations':>12}")
    print("-"*80)
    
    for k in sorted(results_by_k.keys()):
        if results_by_k[k] is not None:
            res = results_by_k[k]
            print(f"{k:>4} {res['mean_mse']:>10.4f} {res['std_mse']:>10.4f} "
                  f"{res['min_mse']:>10.4f} {res['max_mse']:>10.4f} "
                  f"{res['iterations_completed']:>12}")
        else:
            print(f"{k:>4} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12}")


def targeted_cross_validation(model_class, df_train, test_sets_dict, **model_kwargs):
    """
    Perform targeted cross-validation with specific test sets.
    
    For each key-value pair in test_sets_dict, train on all other samples
    and test on the specified samples.
    
    Parameters
    ----------
    model_class : class
        The model class (CorrectionByModel)
    df_train : pd.DataFrame
        Training dataframe with measurements
    test_sets_dict : dict
        Dictionary of test set names to sample numbers
        Example: {'Blues': [1, 2, 3], 'Reds': [4, 5, 6]}
    **model_kwargs : dict
        Keyword arguments for initializing the model
        
    Returns
    -------
    dict : Dictionary with targeted CV results
    """
    print("="*80)
    print("TARGETED CROSS-VALIDATION")
    print("="*80)
    
    print("\nTest Sets Dictionary:")
    for name, samples in test_sets_dict.items():
        print(f"  {name}: {samples}")
    
    # Get all unique sample numbers from dataframe
    all_sample_numbers = df_train['sample_number'].unique().tolist()
    
    targeted_results = {}
    
    # Run one iteration for each key in the dictionary
    for test_set_name, test_sample_numbers in test_sets_dict.items():
        print(f"\n{'='*60}")
        print(f"Iteration for Test Set: {test_set_name}")
        print(f"{'='*60}")
        
        # Ensure test_sample_numbers are integers and exist in dataframe
        test_sample_numbers = [int(s) for s in test_sample_numbers]
        valid_test_samples = [s for s in test_sample_numbers if s in all_sample_numbers]
        
        if not valid_test_samples:
            print(f"Warning: No valid test samples found for {test_set_name}")
            targeted_results[test_set_name] = {
                'mse': np.nan,
                'test_samples': test_sample_numbers,
                'valid_test_samples': [],
                'n_train': 0,
                'n_test': 0,
                'error': 'No valid test samples'
            }
            continue
        
        print(f"Valid test samples: {valid_test_samples}")
        
        # Get training samples (all samples not in test set)
        train_sample_numbers = [s for s in all_sample_numbers if s not in valid_test_samples]
        
        # Create train and test dataframes
        cv_df_train = df_train[df_train['sample_number'].isin(train_sample_numbers)].copy()
        cv_df_test = df_train[df_train['sample_number'].isin(valid_test_samples)].copy()
        
        print(f"Training samples: {len(train_sample_numbers)} unique, {len(cv_df_train)} rows")
        print(f"Test samples: {len(valid_test_samples)} unique, {len(cv_df_test)} rows")
        
        # Check if we have enough training data
        if len(cv_df_train) < 10:
            print(f"Warning: Insufficient training data ({len(cv_df_train)} rows). Skipping.")
            targeted_results[test_set_name] = {
                'mse': np.nan,
                'test_samples': valid_test_samples,
                'n_train': len(cv_df_train),
                'n_test': len(cv_df_test),
                'error': 'Insufficient training data'
            }
            continue
        
        # Initialize and train model
        model = model_class(**model_kwargs)
        model.train(cv_df_train)
        
        # Apply correction to test set
        cv_df_test_corrected = model.apply_correction(cv_df_test.copy())
        
        # Calculate MSE
        avg_mse = calculate_mse_for_model(model, cv_df_test_corrected)
        
        print(f"\nResults for '{test_set_name}':")
        print(f"  Average MSE: {avg_mse:.4f}")
        print(f"  Training size: {len(cv_df_train)} rows")
        print(f"  Test size: {len(cv_df_test)} rows")
        
        targeted_results[test_set_name] = {
            'mse': avg_mse,
            'test_samples': valid_test_samples,
            'n_train': len(cv_df_train),
            'n_test': len(cv_df_test),
            'coeffs': model.coeffs.copy() if hasattr(model, 'coeffs') else None
        }
    
    # Plot results
    plot_targeted_results(targeted_results, save_path="targeted_cv_analysis.png")
    
    # Print summary
    print_targeted_summary(targeted_results)
    
    return targeted_results


def print_targeted_summary(targeted_results):
    """Print summary of targeted cross-validation results."""
    print("\n" + "="*80)
    print("TARGETED CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(f"{'Test Set':<20} {'MSE':<12} {'Train Samples':<15} {'Test Samples':<15} {'Test Samples List'}")
    print("-"*80)
    
    for name, res in targeted_results.items():
        if np.isnan(res['mse']) or 'error' in res:
            status = res.get('error', 'Invalid result')
            print(f"{name:<20} {'N/A':<12} {res.get('n_train', 'N/A'):<15} "
                  f"{res.get('n_test', 'N/A'):<15} {status}")
        else:
            test_samples_str = str(res['test_samples'])[:30] + "..." if len(str(res['test_samples'])) > 30 else str(res['test_samples'])
            print(f"{name:<20} {res['mse']:<12.4f} {res['n_train']:<15} "
                  f"{res['n_test']:<15} {test_samples_str}")


# Example usage function
def run_comprehensive_cross_validation(df_train):
    """
    Run both k-color-out and targeted cross-validation.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe
    """
    
    print("="*100)
    print("COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("="*100)
    
    # 1. K-Color-Out Cross-Validation
    print("\n\n1. K-COLOR-OUT CROSS-VALIDATION")
    print("-"*50)

    k_out_results = cross_validate_model_k_out(
        model_class=CorrectionByModel,
        df_train=df_train,
        k_min=1,
        k_max=20,
        iterations_per_k=20,
        space=ColorSpace.LAB,
        method='joint',
        degree=1,
        pose=True,
        reg_degree=0.0,
        reg_pose=0.0,
        boundary_penalty_factor=0.0001,
        r=4
    )
    
    # k_out_results = cross_validate_model_k_out_fix(
    #     model_class=CorrectionByModel,
    #     df_train=df_train,
    #     k_min=1,
    #     k_max=20,
    #     iterations_per_k=20,
    #     space=ColorSpace.LAB,
    #     method='joint',
    #     degree=1,
    #     pose=True,
    #     reg_degree=0.0,
    #     reg_pose=0.0,
    #     boundary_penalty_factor=0.0001,
    #     r=4
    # )
    
    # 2. Targeted Cross-Validation with default color categories
    # 2. Targeted Cross-Validation with default color categories
    print("\n\n2. TARGETED CROSS-VALIDATION")
    print("-"*50)
    
    # Define default color categories
    default_test_sets = {
        'Red-ish': [7, 9, 12, 15, 16],
        'Green-ish': [4, 6, 11, 14],
        'Blue-ish': [3, 5, 6, 8, 13, 18],
        'Neutral': [19, 20, 21, 22, 23, 24],
        'Top_5_R': df_train.groupby('sample_number')['gt__R'].max().nlargest(5).index.tolist(),
        'Top_5_G': df_train.groupby('sample_number')['gt__G'].max().nlargest(5).index.tolist(),
        'Top_5_B': df_train.groupby('sample_number')['gt__B'].max().nlargest(5).index.tolist()
    }
    
    targeted_results = targeted_cross_validation(
        model_class=CorrectionByModel,
        df_train=df_train,
        test_sets_dict=default_test_sets,
        space=ColorSpace.LAB,
        method='joint',
        degree=1,
        pose=True,
        reg_degree=0.0,
        reg_pose=0.0,
        boundary_penalty_factor=0.0001,
        r=4
    )
    
    # 3. Optional: Extreme RGB targeted validation
    # print("\n\n3. EXTREME RGB TARGETED VALIDATION")
    # print("-"*50)
    
    # # Find extreme RGB samples
    # max_r_samples = df_train.groupby('sample_number')['gt__R'].max().nlargest(5).index.tolist()
    # max_g_samples = df_train.groupby('sample_number')['gt__G'].max().nlargest(5).index.tolist()
    # max_b_samples = df_train.groupby('sample_number')['gt__B'].max().nlargest(5).index.tolist()
    
    # extreme_test_sets = {
    #     'Top_5_R': max_r_samples,
    #     'Top_5_G': max_g_samples,
    #     'Top_5_B': max_b_samples
    # }
    
    # extreme_results = targeted_cross_validation(
    #     model_class=CorrectionByModel,
    #     df_train=df_train,
    #     test_sets_dict=extreme_test_sets,
    #     space=ColorSpace.LAB,
    #     method='joint',
    #     degree=1,
    #     pose=True,
    #     reg_degree=0.0,
    #     reg_pose=0.0,
    #     boundary_penalty_factor=0.0001,
    #     r=4
    # )
    
    return {
        'k_out_results': k_out_results,
        'targeted_results': targeted_results,
        #'extreme_results': extreme_results
    }

def leave_one_color_out_analysis(df_train, model_class, **model_kwargs):
    """
    Perform leave-one-color-out analysis to see which colors cause the biggest loss surge.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe
    model_class : class
        Model class (CorrectionByModel or CorrectionByScaling)
    **model_kwargs : dict
        Model parameters
        
    Returns
    -------
    pd.DataFrame : Results for each left-out color
    """
    print("="*80)
    print("LEAVE-ONE-COLOR-OUT ANALYSIS")
    print("="*80)
    
    unique_sample_numbers = df_train['sample_number'].unique()
    all_sample_numbers = sorted(unique_sample_numbers.tolist())
    
    print(f"Total colors: {len(all_sample_numbers)}")
    print(f"Colors: {all_sample_numbers}")
    print()
    
    results = []
    
    for left_out_color in all_sample_numbers:
        print(f"Leaving out color {left_out_color}...")
        
        # Training data: all colors except left_out_color
        train_samples = [s for s in all_sample_numbers if s != left_out_color]
        test_samples = [left_out_color]
        
        # Split data
        cv_df_train = df_train[df_train['sample_number'].isin(train_samples)].copy()
        cv_df_test = df_train[df_train['sample_number'].isin(test_samples)].copy()
        
        # Initialize and train model
        model = model_class(**model_kwargs)
        model.train(cv_df_train)
        
        # Apply correction to test set
        cv_df_test_corrected = model.apply_correction(cv_df_test.copy())
        
        # Calculate MSE
        avg_mse = calculate_mse_for_model(model, cv_df_test_corrected)
        
        # Get some statistics about this color
        color_stats = df_train[df_train['sample_number'] == left_out_color].iloc[0]
        
        results.append({
            'left_out_color': left_out_color,
            'mse': avg_mse,
            'n_train_samples': len(train_samples),
            'n_test_samples': len(test_samples),
            'gt_R_mean': color_stats['gt__R'],
            'gt_G_mean': color_stats['gt__G'],
            'gt_B_mean': color_stats['gt__B'],
            'color_r4_R_mean': color_stats[f'color_r4_R'],
            'color_r4_G_mean': color_stats[f'color_r4_G'],
            'color_r4_B_mean': color_stats[f'color_r4_B']
        })
        
        print(f"  MSE when leaving out color {left_out_color}: {avg_mse:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by MSE (highest to lowest)
    results_df = results_df.sort_values('mse', ascending=False).reset_index(drop=True)
    
    # Add rank
    results_df['rank'] = range(1, len(results_df) + 1)
    
    print("\n" + "="*80)
    print("ANALYSIS RESULTS (Sorted by MSE, highest first)")
    print("="*80)
    print(f"{'Rank':<5} {'Color':<8} {'MSE':<12} {'GT RGB':<20} {'Measured RGB':<20}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        gt_rgb = f"({row['gt_R_mean']:.0f},{row['gt_G_mean']:.0f},{row['gt_B_mean']:.0f})"
        meas_rgb = f"({row['color_r4_R_mean']:.0f},{row['color_r4_G_mean']:.0f},{row['color_r4_B_mean']:.0f})"
        print(f"{row['rank']:<5} {row['left_out_color']:<8} {row['mse']:<12.4f} {gt_rgb:<20} {meas_rgb:<20}")
    
    # Plot the results
    plot_leave_one_out_results(results_df)
    
    return results_df

def leave_one_color_out_analysis_fix(df_train, model_class, **model_kwargs):
    """
    Perform leave-one-color-out analysis to see which colors cause the biggest loss surge.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe
    model_class : class
        Model class (CorrectionByModel or CorrectionByScaling)
    **model_kwargs : dict
        Model parameters
        
    Returns
    -------
    pd.DataFrame : Results for each left-out color
    """
    print("="*80)
    print("LEAVE-ONE-COLOR-OUT ANALYSIS")
    print("="*80)
    
    unique_sample_numbers = df_train['sample_number'].unique()
    all_sample_numbers = sorted(unique_sample_numbers.tolist())
    
    print(f"Total colors: {len(all_sample_numbers)}")
    print(f"Colors: {all_sample_numbers}")
    print()
    
    rgb_measured = ['color_r4_R', 'color_r4_G', 'color_r4_B']
    lab_ground_truth_cols = ['gt__l', 'gt__a', 'gt__b']
    lab_corrected = ['correction_r4_l', 'correction_r4_a', 'correction_r4_b']

    detailed_point_logs = []

    for left_out_color in all_sample_numbers:
        print(f"Leaving out color {left_out_color}...")
        
        # Training data: all colors except left_out_color
        train_samples = [s for s in all_sample_numbers if s != left_out_color]
        test_samples = [left_out_color]
        
        # Split data
        cv_df_train = df_train[df_train['sample_number'].isin(train_samples)].copy()
        cv_df_test = df_train[df_train['sample_number'].isin(test_samples)].copy()
        
        # Initialize and train model
        model = model_class(**model_kwargs)
        model.train(cv_df_train)
        
        # Apply correction to test set
        cv_df_test_corrected = model.apply_correction(cv_df_test.copy())
        
        # Calculate Euclidean distance (delta E) in LAB space
        diffs = cv_df_test_corrected[lab_corrected].values - cv_df_test_corrected[lab_ground_truth_cols].values
        distances = np.linalg.norm(diffs, axis=1)
        accuracy = np.mean(distances < 2.0)  # Percentage of samples with delta E < 2
        avg_dist = np.mean(distances)

        # # Calculate MSE
        # avg_mse = calculate_mse_for_model(model, cv_df_test_corrected)
        
        # Get some statistics about this color
        #color_stats = df_train[df_train['sample_number'] == left_out_color].iloc[0]
        
        for i, (_, row) in enumerate(cv_df_test_corrected.iterrows()):
            detailed_point_logs.append({
                'left_out_color': left_out_color,
                'sample_id': row['sample_number'],
                'R_measured': row['color_r4_R'],
                'G_measured': row['color_r4_G'],
                'B_measured': row['color_r4_B'],
                'L_corrected': row['correction_r4_l'],
                'a_corrected': row['correction_r4_a'],
                'b_corrected': row['correction_r4_b'],
                'L_ground_truth': row['gt__l'],
                'a_ground_truth': row['gt__a'],
                'b_ground_truth': row['gt__b'],
                'delta_E': distances[i],
                'is_accurate': distances[i] < 2.0
            })
    pd.DataFrame(detailed_point_logs).to_csv('loo_detailed_points.csv', index=False)        
    
    # Create results dataframe
    # results_df = pd.DataFrame()
    
    # # Sort by MSE (highest to lowest)
    # results_df = results_df.sort_values('mse', ascending=False).reset_index(drop=True)
    
    # # Add rank
    # results_df['rank'] = range(1, len(results_df) + 1)
    
    # print("\n" + "="*80)
    # print("ANALYSIS RESULTS (Sorted by MSE, highest first)")
    # print("="*80)
    # print(f"{'Rank':<5} {'Color':<8} {'MSE':<12} {'GT RGB':<20} {'Measured RGB':<20}")
    # print("-"*80)
    
    # for _, row in results_df.iterrows():
    #     gt_rgb = f"({row['gt_R_mean']:.0f},{row['gt_G_mean']:.0f},{row['gt_B_mean']:.0f})"
    #     meas_rgb = f"({row['color_r4_R_mean']:.0f},{row['color_r4_G_mean']:.0f},{row['color_r4_B_mean']:.0f})"
    #     print(f"{row['rank']:<5} {row['left_out_color']:<8} {row['mse']:<12.4f} {gt_rgb:<20} {meas_rgb:<20}")
    
    # # Plot the results
    # # plot_leave_one_out_results(results_df)
    # pd.DataFrame.from_dict(results_df).to_csv('loo_cv.csv', index=False)
    
    # return results_df


# Add this function to your existing code and call it like this:

def run_leave_one_out_analysis(df_train):
    """Run leave-one-color-out analysis."""
    print("\n" + "="*100)
    print("RUNNING LEAVE-ONE-COLOR-OUT ANALYSIS")
    print("="*100)
    
    results = leave_one_color_out_analysis(
        df_train=df_train,
        model_class=CorrectionByModel,
        space=ColorSpace.LAB,
        method='joint',
        degree=1,
        pose=False,
        reg_degree=0.0,
        reg_pose=0.0,
        boundary_penalty_factor=0.0,
        r=4
    )
    
    return results


# Or if you want to compare different models:
def compare_models_leave_one_out(df_train):
    """Compare different models using leave-one-out analysis."""
    
    models_to_test = [
        {
            'name': 'CorrectionByModel (RGB, joint, degree=2)',
            'class': CorrectionByModel,
            'params': {
                'space': ColorSpace.LAB,
                'method': 'joint',
                'degree': 1,
                'pose': False,
                'reg_degree': 0.0,
                'reg_pose': 0.0,
                'boundary_penalty_factor': 0.0,
                'r': 4
            }
        },
        # {
        #     'name': 'CorrectionByModel (LAB, joint, degree=2)',
        #     'class': CorrectionByModel,
        #     'params': {
        #         'space': ColorSpace.LAB,
        #         'method': 'joint',
        #         'degree': 1,
        #         'pose': False,
        #         'reg_degree': 0.0,
        #         'reg_pose': 0.0,
        #         'boundary_penalty_factor': 0.0,
        #         'r': 4
        #     }
        # },
    ]
    
    all_results = {}
    
    for model_config in models_to_test:
        print(f"\n\n{'='*80}")
        print(f"Testing: {model_config['name']}")
        print(f"{'='*80}")
        
        results_df = leave_one_color_out_analysis(
            df_train=df_train,
            model_class=model_config['class'],
            **model_config['params']
        )
        
        all_results[model_config['name']] = {
            'results_df': results_df,
            'mean_mse': results_df['mse'].mean(),
            'max_mse': results_df['mse'].max(),
            'min_mse': results_df['mse'].min(),
            'top_problematic': results_df.head(3)['left_out_color'].tolist()
        }
    
    # Print comparison summary
    print("\n\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Model':<40} {'Avg MSE':<12} {'Max MSE':<12} {'Min MSE':<12} {'Top 3 Problematic Colors'}")
    print("-"*100)
    
    for model_name, results in all_results.items():
        problematic_str = ", ".join([str(c) for c in results['top_problematic']])
        print(f"{model_name:<40} {results['mean_mse']:<12.4f} {results['max_mse']:<12.4f} "
              f"{results['min_mse']:<12.4f} {problematic_str}")
    
    return all_results


if __name__ == "__main__":
    import pickle
    random.seed(0)

    # Load data
    df_daylight1 = load_data("Data/Jonas1.json")
    df_daylight1["lighting_condition"] = LightingCondition.DAYLIGHT.value
    df_daylight2 = load_data("Data/Baisu1.json")
    df_daylight2["lighting_condition"] = LightingCondition.DAYLIGHT.value
    df_raw = pd.concat([df_daylight1, df_daylight2], ignore_index=True)
    for prefix in ["color_r4_", "gt__"]:
        df_raw = convert_rgb_cols(df_raw, prefix, to=ColorSpace.LAB)

    # Train-test split
    df_train = df_raw.sample(frac=0.8, random_state=42)
    training_indices = df_train.index
    df_test = df_raw.drop(training_indices)

    # Leave k out
    # 1. K-Color-Out Cross-Validation
    print("\n\n1. K-COLOR-OUT CROSS-VALIDATION")
    print("-"*50)
    
    k_out_results = cross_validate_model_k_out(
        model_class=CorrectionByModel,
        df_train=df_train,
        k_min=1,
        k_max=20,
        iterations_per_k=20,
        space=ColorSpace.LAB,
        method='joint',
        degree=1,
        pose=False,
        reg_degree=0.0,
        reg_pose=0.0,
        boundary_penalty_factor=0.0,
        r=4
    )
    pickle.dump(k_out_results, open("k_out_results.pkl", "wb"))

    # Leave one out analysis
    leave_one_out_results = leave_one_color_out_analysis(
        df_train=df_train,
        model_class=CorrectionByModel,
        space=ColorSpace.LAB,
        method='joint',
        degree=1,
        pose=False,
        reg_degree=0.0,
        reg_pose=0.0,
        boundary_penalty_factor=0.0,
        r=4
    )
    pickle.dump(leave_one_out_results, open("leave_one_out_results.pkl", "wb"))


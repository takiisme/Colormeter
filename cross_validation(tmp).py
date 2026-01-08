import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error


def cross_validate_model(model, df_train, k=5):
    unique_sample_numbers = df_train['sample_number'].unique()
    print(f"Unique sample numbers: {unique_sample_numbers}")
    print(f"Number of unique sample numbers: {len(unique_sample_numbers)}")

    # Initialize a dictionary to store MSE results for each k
    mse_results_by_k = {}

    # Get the unique sample numbers
    all_sample_numbers = unique_sample_numbers.tolist()
    num_total_samples = len(all_sample_numbers)

    # Iterate through k values from 1 to 10
    for k in range(1, 21):
        print(f"\n--- Running k-Color-Out Cross-Validation for k = {k} ---")
        iteration_mses = []

        # Perform 10 random iterations for each k
        for iteration in range(20):
            # a. Randomly select k unique sample numbers for testing
            test_samples = random.sample(all_sample_numbers, k)
            print(test_samples)

            # b. Use the remaining 24 - k unique sample numbers for training
            train_samples = [s for s in all_sample_numbers if s not in test_samples]

            # c. Create cv_df_train by filtering df_train
            cv_df_train = df_train[df_train['sample_number'].isin(train_samples)].copy()

            # d. Create cv_df_test by filtering df_train
            cv_df_test = df_train[df_train['sample_number'].isin(test_samples)].copy()

            # Ensure cv_df_train is not empty before training
            if cv_df_train.empty:
                print(f"Warning: Training set is empty for k={k}, iteration={iteration}. Skipping.")
                continue

            # e. Train the polynomial correction model
            # Only use the 'color_r4_' prefix for training as per the original model
            train_results = fit_rgb_polynomial_with_lighting(cv_df_train, meas_prefix="color_r4_")

            best_coeffs_R = train_results["R"]["theta"]
            best_coeffs_G = train_results["G"]["theta"]
            best_coeffs_B = train_results["B"]["theta"]

            # f. Apply corrections to the test set
            # Create a copy to avoid SettingWithCopyWarning
            cv_df_test_corrected = cv_df_test.copy()
            cv_df_test_corrected = correctRGB(
                cv_df_test_corrected,
                correction_type='polynomial',
                coeffs_R=best_coeffs_R,
                coeffs_G=best_coeffs_G,
                coeffs_B=best_coeffs_B
            )

            # g. Calculate the average Mean Squared Error (MSE) for the test set
            if not cv_df_test_corrected.empty:
                mse_r = mean_squared_error(cv_df_test_corrected['gt__R'], cv_df_test_corrected['correction_r4_R'])
                mse_g = mean_squared_error(cv_df_test_corrected['gt__G'], cv_df_test_corrected['correction_r4_G'])
                mse_b = mean_squared_error(cv_df_test_corrected['gt__B'], cv_df_test_corrected['correction_r4_B'])

                avg_mse = (mse_r + mse_g + mse_b) / 3.0
                iteration_mses.append(avg_mse)
                print(f"k={k}, Iteration {iteration+1}: Avg MSE = {avg_mse:.2f}")
            else:
                print(f"Warning: Test set is empty for k={k}, iteration={iteration}. Skipping MSE calculation.")

        # h. Store the calculated average MSEs for the current k
        if iteration_mses:
            mse_results_by_k[k] = iteration_mses
            print(f"Finished k={k} with {len(iteration_mses)} iterations. Average MSE for k={k}: {np.mean(iteration_mses):.2f}")
        else:
            print(f"No MSE results recorded for k={k}.")

    print("\n--- Cross-Validation Complete ---")
    print("MSE results by k:", mse_results_by_k)

    # Calculate mean and standard deviation for each k
    mean_mses = {k: np.mean(mses) for k, mses in mse_results_by_k.items()}
    std_mses = {k: np.std(mses) for k, mses in mse_results_by_k.items()}

    k_values = sorted(mean_mses.keys())
    mean_values = [mean_mses[k] for k in k_values]
    std_values = [std_mses[k] for k in k_values]

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, mean_values, yerr=std_values, fmt='-o', capsize=5, capthick=2, ecolor='red', color='blue', linewidth=2)
    plt.title('Average MSE vs. Number of Test Samples (k-Color-Out Cross-Validation)')
    plt.xlabel('Number of Test Samples (k)')
    plt.ylabel('Average MSE')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    # Summarize the model's stability and performance
    print("\n--- Cross-Validation Summary ---")
    for k in k_values:
        print(f"k = {k}: Average MSE = {mean_mses[k]:.2f}, Std Dev MSE = {std_mses[k]:.2f}")


def cross_validation_leave_k_out_on_purpose(model, df_train, k=5):
    color_categories = {
        'Blue-ish': [3, 6, 8, 13, 18],
        'Green-ish': [4, 6, 11, 14],
        'Neutral': [19, 20, 21, 22, 23, 24],
        'Red-ish': [7, 9, 12, 15]
    }

    print("Color Categories Dictionary:")
    print(color_categories)

    # Get unique samples with their max RGB values
    max_r_samples = df_train.groupby('sample_number')['gt__R'].max().nlargest(5).index.tolist()
    max_g_samples = df_train.groupby('sample_number')['gt__G'].max().nlargest(5).index.tolist()
    max_b_samples = df_train.groupby('sample_number')['gt__B'].max().nlargest(5).index.tolist()

    extreme_rgb_samples = {
        'Top_5_largest_R': max_r_samples,
        'Top_5_largest_G': max_g_samples,
        'Top_5_largest_B': max_b_samples
    }

    print("Extreme RGB Samples:")
    print(extreme_rgb_samples)

    all_test_sets = {**color_categories, **extreme_rgb_samples}

    targeted_mse_results = {}

    # Get all unique sample numbers from the training dataframe
    all_sample_numbers = df_train['sample_number'].unique().tolist()

    for test_set_name, test_sample_numbers in all_test_sets.items():
        print(f"\n--- Running Targeted Cross-Validation for: {test_set_name} ---")
        iteration_mses = []

        # Ensure test_sample_numbers are integers
        test_sample_numbers = [int(s) for s in test_sample_numbers]

        # The test set consists of all rows corresponding to the current test_sample_numbers
        cv_df_test = df_train[df_train['sample_number'].isin(test_sample_numbers)].copy()

        # The training set consists of all rows NOT corresponding to the current test_sample_numbers
        train_sample_numbers = [s for s in all_sample_numbers if s not in test_sample_numbers]
        cv_df_train = df_train[df_train['sample_number'].isin(train_sample_numbers)].copy()

        # Ensure training set is not empty
        if cv_df_train.empty:
            print(f"Warning: Training set is empty for {test_set_name}. Skipping.")
            targeted_mse_results[test_set_name] = {'mse': np.nan, 'std': np.nan, 'iterations': 0}
            continue

        # Train the polynomial correction model on the training set
        # Only use the 'color_r4_' prefix for training
        train_results = fit_rgb_polynomial_with_lighting(cv_df_train, meas_prefix="color_r4_")

        best_coeffs_R = train_results["R"]["theta"]
        best_coeffs_G = train_results["G"]["theta"]
        best_coeffs_B = train_results["B"]["theta"]

        # Apply corrections to the test set
        if not cv_df_test.empty:
            cv_df_test_corrected = cv_df_test.copy()
            cv_df_test_corrected = correctRGB(
                cv_df_test_corrected,
                correction_type='polynomial',
                coeffs_R=best_coeffs_R,
                coeffs_G=best_coeffs_G,
                coeffs_B=best_coeffs_B
            )

            # Calculate the average Mean Squared Error (MSE) for the test set
            mse_r = mean_squared_error(cv_df_test_corrected['gt__R'], cv_df_test_corrected['correction_r4_R'])
            mse_g = mean_squared_error(cv_df_test_corrected['gt__G'], cv_df_test_corrected['correction_r4_G'])
            mse_b = mean_squared_error(cv_df_test_corrected['gt__B'], cv_df_test_corrected['correction_r4_B'])

            avg_mse = (mse_r + mse_g + mse_b) / 3.0
            iteration_mses.append(avg_mse) # Store the average MSE for this test set
            print(f"Average MSE for {test_set_name} test set: {avg_mse:.2f}")
        else:
            print(f"Warning: Test set is empty for {test_set_name}. Skipping MSE calculation.")

        # Store the results for the current test set
        if iteration_mses:
            targeted_mse_results[test_set_name] = {
                'mse': np.mean(iteration_mses),
                'std': np.std(iteration_mses) if len(iteration_mses) > 1 else 0.0,
                'iterations': len(iteration_mses)
            }
        else:
            targeted_mse_results[test_set_name] = {'mse': np.nan, 'std': np.nan, 'iterations': 0}

    print("\n--- Targeted Cross-Validation Complete ---")
    print("Targeted MSE Results:")
    for name, res in targeted_mse_results.items():
        print(f"{name}: Avg MSE = {res['mse']:.2f}, Std Dev = {res['std']:.2f}")

        import matplotlib.pyplot as plt

    # Prepare data for plotting
    plot_data = {
        'Test Set': [],
        'Average MSE': []
    }

    for name, res in targeted_mse_results.items():
        plot_data['Test Set'].append(name)
        plot_data['Average MSE'].append(res['mse'])

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Test Set', y='Average MSE', data=plot_data, palette='viridis')
    plt.title('Average MSE for Targeted Leave-k-out Cross-Validation (r4)')
    plt.xlabel('Test Set Category')
    plt.ylabel('Average MSE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n--- Summary of Targeted Cross-Validation Findings ---")
    for name, res in targeted_mse_results.items():
        print(f"Test Set: {name.ljust(20)} | Average MSE: {res['mse']:.2f} | Std Dev MSE: {res['std']:.2f}")
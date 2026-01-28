import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from util import load_data
from plot import plot_comparison_grid, plot_against_gt, plotHSV, plotHSVError, plot_ecdf
from correction import CorrectionByModel, CorrectionByScaling
from color_conversion import convert_rgb_cols, convert_to_rgb
from constants import ColorSpace
from cross_validation import *

df_daylight1 = load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = "daylight"
df_dark1 = load_data("Data/Tai4.json")
df_dark1["lighting_condition"] = "dark"
df_dark2 = load_data("Data/Zhi3.json")
df_dark2["lighting_condition"] = "dark"

#-------------------------------------------------------------------------------
# df = pd.concat([df_daylight1, df_daylight2, df_dark1, df_dark2], ignore_index=True)
# corrector0, corrector2, corrector4 = CorrectionByScaling(space=ColorSpace.RGB, r=0), CorrectionByScaling(space=ColorSpace.RGB, r=2), CorrectionByScaling(space=ColorSpace.RGB, r=4)
# df_corrected = corrector4.predict(corrector2.predict(corrector0.predict(df)))
# fig, ax = plot.plot_comparison_grid(df_corrected, radius=4, rows=4, cols=6)
# #fig, ax = plot.plotHSV(df_corrected)
# #fig, ax = plot.plotHSVError(df_corrected)
#-------------------------------------------------------------------------------

df = pd.concat([df_daylight1, df_daylight2], ignore_index=True)
df_raw = df.copy()
df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.LAB)
df = convert_rgb_cols(df, prefix="color_r4_", to=ColorSpace.LAB)
df_train = df.sample(frac=0.66, random_state=0)
df_test = df.drop(df_train.index)


corrector_scaling = CorrectionByScaling(space=ColorSpace.RGB, r=4)
df_test = corrector_scaling.apply_correction(df_test, prefix="scaling_correction")
df_test = convert_rgb_cols(df_test, prefix="scaling_correction_r4_", to=ColorSpace.LAB)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].set_title("RGB Euclidean Error ECDF")
axs[0].set_xlabel("Euclidean Error")
axs[1].set_title("Lab Euclidean Error ECDF")
axs[1].set_xlabel("Euclidean Error")

eucl_raw_rgb = np.sqrt(
    (df_test["color_r4_R"] - df_test["gt__R"])**2 + \
    (df_test["color_r4_G"] - df_test["gt__G"])**2 + \
    (df_test["color_r4_B"] - df_test["gt__B"])**2
)
axs[0] = plot_ecdf(axs[0], eucl_raw_rgb, label="raw", color='blue')
eucl_scaling_rgb = np.sqrt(
    (df_test["scaling_correction_r4_R"] - df_test["gt__R"])**2 + \
    (df_test["scaling_correction_r4_G"] - df_test["gt__G"])**2 + \
    (df_test["scaling_correction_r4_B"] - df_test["gt__B"])**2
)
axs[0] = plot_ecdf(axs[0], eucl_scaling_rgb, label="scaling in RGB", color='red')
eucl_raw_lab = np.sqrt(
    (df_test["color_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["color_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["color_r4_b"] - df_test["gt__b"])**2
)
axs[1] = plot_ecdf(axs[1], eucl_raw_lab, label="raw", color='blue')
eucl_scaling_lab = np.sqrt(
    (df_test["scaling_correction_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["scaling_correction_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["scaling_correction_r4_b"] - df_test["gt__b"])**2
)
axs[1] = plot_ecdf(axs[1], eucl_scaling_lab, label="scaling in RGB", color='red')

# Model
corrector_model = CorrectionByModel(space=ColorSpace.LAB, r=4, degree=1, boundary_penalty_factor=0, pose=False)
corrector_model.train(df_train) # Comment this line out and uncomment the block below leads to bootstrapping on model's weights

##############################################################################
# Train the model with alpha = 0.05
# corrector_model.train_with_bootstrap(df_train, n_iterations=1000, alpha=0.05)
# np.save("bootstrapped_coeffs_pose=False.npz", np.array(corrector_model.bootstrapped_coeffs))

# # Flatten coefficients for file export
# # We convert the list of arrays into one large matrix
# boot_data = np.array(corrector_model.bootstrapped_coeffs)
# n_iters = boot_data.shape[0]
# flattened = boot_data.reshape(n_iters, -1)

# # Calculate final statistics
# means = np.mean(flattened, axis=0)
# ci_lower = np.percentile(flattened, 2.5, axis=0)
# ci_upper = np.percentile(flattened, 97.5, axis=0)

# # Create the summary DataFrame
# param_names = [f"Beta_{i}" for i in range(flattened.shape[1])]
# summary_df = pd.DataFrame({
#     'parameter': param_names,
#     'mean': means,
#     'ci_025': ci_lower, # 2.5th percentile
#     'ci_975': ci_upper  # 97.5th percentile
# })

# # Export to file
# summary_df.to_csv("model_parameters_95CI.csv", index=False)
# print("Saved 95% Confidence Intervals to model_parameters_95CI.csv")


#############################################################################
df_test = corrector_model.apply_correction(df_test, prefix="model_correction")
# Something that we probably shouldn't do: apply correction to the whole test set at once.
# df_test = df
# df_test = corrector_model.apply_correction(df_test, prefix="model_correction")

df_test = convert_rgb_cols(df_test, prefix="gt__", to=ColorSpace.LAB)
df_test = convert_to_rgb(df_test, prefix="model_correction_r4_", from_space=ColorSpace.LAB)

eucl_model_rgb = np.sqrt(
    (df_test["model_correction_r4_R"] - df_test["gt__R"])**2 + \
    (df_test["model_correction_r4_G"] - df_test["gt__G"])**2 + \
    (df_test["model_correction_r4_B"] - df_test["gt__B"])**2
)
axs[0] = plot_ecdf(axs[0], eucl_model_rgb, label="model in Lab", color='purple')
eucl_model_lab = np.sqrt(
    (df_test["model_correction_r4_l"] - df_test["gt__l"])**2 + \
    (df_test["model_correction_r4_a"] - df_test["gt__a"])**2 + \
    (df_test["model_correction_r4_b"] - df_test["gt__b"])**2
)
axs[1] = plot_ecdf(axs[1], eucl_model_lab, label="model in Lab", color='purple')

axs[0].legend()
axs[1].legend()
fig.savefig("error_ecdf_comparison.png")
run_comprehensive_cross_validation(df_raw)
run_leave_one_out_analysis(df_raw)

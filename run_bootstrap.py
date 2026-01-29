import numpy as np
import pandas as pd
from color_conversion import convert_rgb_cols
from constants import ColorSpace
from correction import CorrectionByModel
from util import load_data

df_daylight1 = load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = "daylight"
df = pd.concat([df_daylight1, df_daylight2], ignore_index=True)
df_raw = df.copy()
df = convert_rgb_cols(df, prefix="gt__", to=ColorSpace.LAB)
df = convert_rgb_cols(df, prefix="color_r4_", to=ColorSpace.LAB)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

corrector_model = CorrectionByModel(space=ColorSpace.LAB, r=4, degree=1, boundary_penalty_factor=0, pose=True)
corrector_model.train_with_bootstrap(df_train, n_iterations=1000, alpha=0.05)
np.save('boot_coeffs_pose=True.npy', corrector_model.bootstrapped_coeffs)

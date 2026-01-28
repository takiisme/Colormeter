import numpy as np
import pandas as pd

from color_conversion import convert_rgb_cols
from constants import ColorSpace, LightingCondition
from correction import CorrectionByModel
from util import load_data

corrector_model = CorrectionByModel(space=ColorSpace.LAB, r=4, degree=1, boundary_penalty_factor=0, pose=True)

df = load_data("Data/Jonas1.json")
df["lighting_condition"] = LightingCondition.DAYLIGHT.value
for prefix in ["color_r4_", "gt__", "white_r4_"]:
    df = convert_rgb_cols(df, prefix, to=ColorSpace.LAB)

corrector_model.train(df.copy().iloc[:2,:])
print(corrector_model.build_design_matrix())
print(df[['pitch', 'roll', 'color_r4_l', 'color_r4_a', 'color_r4_b', 'white_r4_l', 'white_r4_a', 'white_r4_b']].iloc[:2,:])

# boot_data = np.load('bootstrapped_coeffs_pose=True.npz.npy')
# print(boot_data[0])

# vars = [
#     'offset_l', 'offset_a', 'offset_b',

# ]
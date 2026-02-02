import pandas as pd
from correction import CorrectionByScaling, CorrectionByModel
from color_conversion import convert_rgb_cols
from constants import ColorSpace, LightingCondition
from util import load_data

# Load data
df_daylight1 = load_data("Data/daylight1.json")
df_daylight1["lighting_condition"] = LightingCondition.DAYLIGHT.value
df_daylight2 = load_data("Data/daylight2.json")
df_daylight2["lighting_condition"] = LightingCondition.DAYLIGHT.value
df_raw = pd.concat([df_daylight1, df_daylight2], ignore_index=True)
for prefix in ["color_r4_", "gt__"]:
    df_raw = convert_rgb_cols(df_raw, prefix, to=ColorSpace.LAB)

# Train-test split
df_train = df_raw.sample(frac=0.8, random_state=42)
training_indices = df_train.index
df_test = df_raw.drop(training_indices)

# Apply corrections
corrector_scaling = CorrectionByScaling(space=ColorSpace.RGB)
df_test = corrector_scaling.apply_correction(df_test.copy(), prefix='scaling')
df_test = convert_rgb_cols(df_test, 'scaling_r4_', to=ColorSpace.LAB)

corrector_model_full = CorrectionByModel(space=ColorSpace.LAB, boundary_penalty_factor=0.0, pose=True)
corrector_model_full.train(df_train.copy())
df_test = corrector_model_full.apply_correction(df_test.copy(), prefix='full')

corrector_model_reduced = CorrectionByModel(space=ColorSpace.LAB, boundary_penalty_factor=0.0, pose=False)
corrector_model_reduced.train(df_train.copy())
df_test = corrector_model_reduced.apply_correction(df_test.copy(), prefix='reduced')

df_test.to_csv("Data/test_corrected_new.csv", index=False)

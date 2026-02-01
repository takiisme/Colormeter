import numpy as np
import pandas as pd
from correction import CorrectionByModel
from color_conversion import convert_rgb_cols
from constants import ColorSpace, LightingCondition
from util import load_data

# Load data
df_daylight1 = load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = LightingCondition.DAYLIGHT.value
df_daylight2 = load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = LightingCondition.DAYLIGHT.value
df_raw = pd.concat([df_daylight1, df_daylight2], ignore_index=True)
for prefix in ["color_r4_", "gt__"]:
    df_raw = convert_rgb_cols(df_raw, prefix, to=ColorSpace.LAB)

df_train = df_raw.sample(frac=0.8, random_state=42)
training_indices = df_train.index
df_test = df_raw.drop(training_indices)

corrector_model_full = CorrectionByModel(space=ColorSpace.LAB, boundary_penalty_factor=0.0, pose=True)
corrector_model_full.train(df_train.copy())
df_test = corrector_model_full.apply_correction(df_test.copy(), prefix='full')

boot_data = np.load('boot_coeffs_pose=True.npy', allow_pickle=True)

coefs = corrector_model_full.coeffs
ci_low = np.percentile(boot_data, 2.5, axis=0)
ci_high = np.percentile(boot_data, 97.5, axis=0)

# Order of coefficients: offset, ref_b, ref_a, ref_l, meas_b, meas_a, meas_l, pitch, roll
param_names = [
    'offset',
    'ref_b', 'ref_a', 'ref_l',
    'meas_b', 'meas_a', 'meas_l',
    'pitch', 'roll'
]
table = pd.DataFrame.from_records(coefs, index=param_names, columns=['corr_l', 'corr_a', 'corr_b']).T
ci_low_df = pd.DataFrame.from_records(ci_low, index=param_names, columns=['corr_l', 'corr_a', 'corr_b']).T
ci_high_df = pd.DataFrame.from_records(ci_high, index=param_names, columns=['corr_l', 'corr_a', 'corr_b']).T

# Combine coefficients and CIs into a single table
for param in param_names:
    ci_low_df[param] = ci_low_df[param].apply(lambda x: f"{x:.2f}")
    ci_high_df[param] = ci_high_df[param].apply(lambda x: f"{x:.2f}")
    for channel in ['corr_l', 'corr_a', 'corr_b']:
        table.at[channel, param] = f"{table.at[channel, param]:.2f} [{ci_low_df.at[channel, param]}, {ci_high_df.at[channel, param]}]"

# Reorder columns
table = table[[
    'offset',
    'meas_l', 'meas_a', 'meas_b',
    'ref_l', 'ref_a', 'ref_b',
    'pitch', 'roll'
]]
# Rename to latex
table.columns = [
    '(offset)',
    r'$L^*_\textup{m}$', r'$a^*_\textup{m}$', r'$b^*_\textup{m}$',
    r'$L^*_\textup{w}$', r'$a^*_\textup{w}$', r'$b^*_\textup{w}$',
    'pitch', 'roll'
]
table.index = [
    r'$\widehat{L^*}$', r'$\widehat{a^*}$', r'$\widehat{b^*}$'
]
table.to_latex('boot_table.tex', escape=False)

from data import load_data, assert_cols_exist
from correction import CorrectionByScaling
from color_conversion import convert_rgb_cols
import pandas as pd
import numpy as np
from constants import ColorSpace, LightingCondition
import seaborn as sns
import matplotlib.pyplot as plt

df_daylight1 = load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = "daylight"
df_dark1 = load_data("Data/Tai4.json")
df_dark1["lighting_condition"] = "dark"
df_dark2 = load_data("Data/Zhi3.json")
df_dark2["lighting_condition"] = "dark"

df = pd.concat([df_daylight1, df_daylight2, df_dark1, df_dark2], ignore_index=True)

corrector = CorrectionByScaling(space=ColorSpace.RGB, r=4)
df_corrected = corrector.predict(df)
df_corrected.to_csv("df_corrected_rgb.csv", index=False)
exit()

df_corrected = convert_rgb_cols(df_corrected, prefix="color_r4_", to=ColorSpace.HSV)
df_corrected = convert_rgb_cols(df_corrected, prefix="correction_r4_", to=ColorSpace.HSV)
df_corrected = convert_rgb_cols(df_corrected, prefix="gt__", to=ColorSpace.HSV)

df_corrected = convert_rgb_cols(df_corrected, prefix="color_r4_", to=ColorSpace.LAB)
df_corrected = convert_rgb_cols(df_corrected, prefix="correction_r4_", to=ColorSpace.LAB)
df_corrected = convert_rgb_cols(df_corrected, prefix="gt__", to=ColorSpace.LAB)


# TEST CODE HERE
from plot import plot_against_gt

g = plot_against_gt(df_corrected, space=ColorSpace.RGB, lighting_condition=LightingCondition.DAYLIGHT, r=4)
g.figure.suptitle('daylight, RGB', y=1.02)
plt.savefig('daylight_rgb.png')

g = plot_against_gt(df_corrected, space=ColorSpace.HSV, lighting_condition=LightingCondition.DAYLIGHT, r=4)
g.figure.suptitle('daylight, HSV', y=1.02)
plt.savefig('daylight_hsv.png')

g = plot_against_gt(df_corrected, space=ColorSpace.LAB, lighting_condition=LightingCondition.DAYLIGHT, r=4)
g.figure.suptitle('daylight, Lab', y=1.02)
plt.savefig('daylight_lab.png')

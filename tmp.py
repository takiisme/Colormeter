from data import load_data
from correction import CorrectionByScaling
import pandas as pd
import numpy as np
from constants import ColorSpace

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

print(df_corrected.head())
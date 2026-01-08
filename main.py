import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import data
import plot
from correction import CorrectionByScaling
from constants import ColorSpace

# TODO: Remove this before submission.
# if __name__ == "__main__":
df_daylight1 = data.load_data("Data/Jonas1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = data.load_data("Data/Baisu1.json")
df_daylight2["lighting_condition"] = "daylight"
df_dark1 = data.load_data("Data/Tai4.json")
df_dark1["lighting_condition"] = "dark"
df_dark2 = data.load_data("Data/Zhi3.json")
df_dark2["lighting_condition"] = "dark"

df = pd.concat([df_daylight1, df_daylight2, df_dark1, df_dark2], ignore_index=True)
corrector0, corrector2, corrector4 = CorrectionByScaling(space=ColorSpace.RGB, r=0), CorrectionByScaling(space=ColorSpace.RGB, r=2), CorrectionByScaling(space=ColorSpace.RGB, r=4)
df_corrected = corrector4.predict(corrector2.predict(corrector0.predict(df)))
#df = plot.plot_comparison_grid(df_corrected, radius=0, rows=4, cols=6)
plot.HSV_error(df)
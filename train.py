# import numpy as np
# import pandas as pd

# from data import load_data
# from constants import ColorSpace, LightingCondition
# from correction import CorrectionByModel

# df_daylight1 = load_data("Data/Jonas1.json")
# df_daylight1["lighting_condition"] = "daylight"
# df_daylight2 = load_data("Data/Baisu1.json")
# df_daylight2["lighting_condition"] = "daylight"
# df_dark1 = load_data("Data/Tai4.json")
# df_dark1["lighting_condition"] = "dark"
# df_dark2 = load_data("Data/Zhi3.json")
# df_dark2["lighting_condition"] = "dark"

# df = pd.concat([df_daylight1, df_daylight2, df_dark1, df_dark2], ignore_index=True)

# df_train = df.sample(frac=0.8, random_state=42)
# df_test = df.drop(df_train.index).reset_index(drop=True)
# corrector = CorrectionByModel(space=ColorSpace.RGB, method='joint', degree=2)
# coefs = corrector.train(df_train)
# # print(coefs.shape)
# df_test = corrector.predict(df_test)
# df_test.to_csv('df_test.csv', index=False)

from plot import plot_comparison_grid, plotHSV, plotHSVError
import matplotlib.pyplot as plt
import pandas as pd
df_test = pd.read_csv('df_test.csv')
fig, ax = plot_comparison_grid(df_test, radius=4, rows=4, cols=6)
plt.savefig('tmp.png')

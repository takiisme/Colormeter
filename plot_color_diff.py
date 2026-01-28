import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import load_data
from correction import CorrectionByModel, CorrectionByScaling
from color_conversion import convert_rgb_cols
from constants import ColorSpace, GT
from skimage.color import rgb2lab, lab2rgb

from tueplots.bundles import icml2024
from tueplots.constants.color import rgb
nrows, ncols = 1, 1
plt.rcParams.update(icml2024(column='half', nrows=nrows, ncols=ncols))

colors = pd.DataFrame(GT).query(
    "label in ['Moderate Red', 'Blue Flower', 'Yellow']"
)[['gt__R', 'gt__G', 'gt__B']].to_numpy() / 255.0
# print(colors)
colors0 = colors.copy()
colors = rgb2lab(colors)

# Perturbation to Delta E == 2
np.random.seed(0)
v = np.random.randn(*colors.shape)
v = v / np.linalg.norm(v, axis=1, keepdims=True) * 2.0
colors_perturbed = colors + v
colors2 = lab2rgb(colors_perturbed)
colors2 = np.clip(colors2, 0.0, 1.0)

# Perturbation to Delta E == 10
np.random.seed(10)
v = np.random.randn(*colors.shape)
v = v / np.linalg.norm(v, axis=1, keepdims=True) * 6.0
colors_perturbed = colors + v
colors6 = lab2rgb(colors_perturbed)
colors6 = np.clip(colors6, 0.0, 1.0)

# Visualize the colors via imshow
fig, ax = plt.subplots(nrows, ncols)

ax.imshow(
    np.concatenate([
        colors0[np.newaxis, :, :],
        colors2[np.newaxis, :, :],
        colors6[np.newaxis, :, :]
    ], axis=0),
    aspect='auto'
)

plt.savefig("plot_color_diff.pdf")

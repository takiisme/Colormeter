from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab
from skimage.color import lab2rgb
from matplotlib.colors import hsv_to_rgb
import numpy as np

# TODO: Allow for conversion from any source (rgb, hsv, lab) to any target (rgb, hsv, lab).
def convert_rgb_cols(df, prefix="color_r4_", to="hsv"):
    """
    Convert RGB columns in a DataFrame to HSV or Lab color space.

    Parameters:
    - prefix: str, the prefix of the RGB columns to convert, e.g., 'color_r4_' or 'gt__'.
    - to: str, target color space, either 'hsv' or 'lab'.

    Returns:
    - A function that takes a DataFrame and returns it with new color space columns added.
    """
    rgb_cols = [f"{prefix}R", f"{prefix}G", f"{prefix}B"]
    rgb_values = df[rgb_cols].to_numpy()
    if any(rgb_values.flatten() > 1):
        rgb_values = rgb_values / 255.0

    if to == "hsv":
        # rgb_to_hsv requires normalized rgb as input and gives normalized hsv as output
        hsv_values = rgb_to_hsv(rgb_values)
        df[f"{prefix}H"] = hsv_values[:, 0]
        df[f"{prefix}S"] = hsv_values[:, 1]
        df[f"{prefix}V"] = hsv_values[:, 2]
    elif to == "lab":
        lab_values = rgb2lab(rgb_values)
        df[f"{prefix}L"] = lab_values[:, 0]
        df[f"{prefix}a"] = lab_values[:, 1]
        df[f"{prefix}b"] = lab_values[:, 2]
    else:
        raise ValueError("Unsupported color space. Use 'hsv' or 'lab'.")

    return df

def convert_to_rgb(df, prefix="color_r4_", from_space="hsv"):
    """
    Convert HSV or Lab columns back to RGB.

    Output RGB is in [0, 255].

    Parameters:
    - df: pandas DataFrame
    - prefix: column prefix, e.g. 'color_r4_' or 'gt__'
    - from_space: 'hsv' or 'lab'
    """

    if from_space == "hsv":
        cols = [f"{prefix}H", f"{prefix}S", f"{prefix}V"]
        hsv = df[cols].to_numpy(dtype=float)

        # H is periodic
        hsv[:, 0] = np.mod(hsv[:, 0], 1.0)

        # hsv_to_rgb expects [0,1]
        rgb = hsv_to_rgb(hsv)

    elif from_space == "lab":
        cols = [f"{prefix}l", f"{prefix}a", f"{prefix}b"]
        lab = df[cols].to_numpy(dtype=float)

        # lab2rgb expects shape (N, 1, 3)
        rgb = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)

    else:
        raise ValueError("from_space must be 'hsv' or 'lab'")

    # clip and scale to [0,255]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).round()

    df[f"{prefix}R"] = rgb[:, 0]
    df[f"{prefix}G"] = rgb[:, 1]
    df[f"{prefix}B"] = rgb[:, 2]

    return df
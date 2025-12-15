from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab

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

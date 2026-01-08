import numpy as np
import pandas as pd


class CorrectionByScaling:
    def __init__(self, space: str = 'rgb'):
        pass




def apply_correction(
        row: pd.Series,
        r: int = 4,
        correction_type: str = 'scaling',
        color_space: str = 'rgb',
        coeffs_H=None,
        coeffs_S=None,
        coeffs_V=None,
        ref_white_H=None,
        ref_white_S=None,
        ref_white_V=None
    ) -> pd.Series:
    """
    Helper function to be passed to DataFrame.apply for color correction.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing measurement and ground-truth data.
    r : int
        Reticle radius (0, 2, or 4) to apply correction on.
    correction_type : str
        Type of correction to apply ('scaling' or 'polynomial').
        TODO: Re-implement polynomial correction.
    TODO: other parameters for polynomial correction
    
    """
    if correction_type == 'scaling':
        assert color_space == 'rgb', "Scaling correction only implemented for RGB space."
        ref_r = row[f'white_r{r}_R']
        ref_g = row[f'white_r{r}_G']
        ref_b = row[f'white_r{r}_B']
        meas_r = row[f'color_r{r}_R']
        meas_g = row[f'color_r{r}_G']
        meas_b = row[f'color_r{r}_B']
        corrected = correct_by_scaling(ref_r, ref_g, ref_b, meas_r, meas_g, meas_b)
    elif correction_type == 'polynomial':
        # corr_h = correctByPolynomial(meas_h, coeffs_H)
        # corr_s = correctByPolynomial(meas_s, coeffs_S)
        # corr_v = correctByPolynomial(meas_v, coeffs_V)
        # # Apply clipping here as correctByPolynomial doesn't do it
        # corr_h = corr_h % 1.0
        # corr_s = np.clip(corr_s, 0, 1.0)
        # corr_v = np.clip(corr_v, 0, 1.0)
        raise NotImplementedError("Polynomial correction TODO.")
    else:
        raise ValueError(f"Unknown correction_type: {correction_type}")

    return pd.Series(corrected)


def correct_by_scaling(ref_r: int, ref_g: int, ref_b: int, meas_r: int, meas_g: int, meas_b: int):
    scale_r = 255.0 / (ref_r if ref_r > 0 else 1.0)
    scale_g = 255.0 / (ref_g if ref_g > 0 else 1.0)
    scale_b = 255.0 / (ref_b if ref_b > 0 else 1.0)

    corr_r = np.clip(meas_r * scale_r, 0, 255).astype(int)
    corr_g = np.clip(meas_g * scale_g, 0, 255).astype(int)
    corr_b = np.clip(meas_b * scale_b, 0, 255).astype(int)
    return corr_r, corr_g, corr_b

def correct_by_polynomial():
    # TODO: Re-implement polynomial correction
    pass


# TODO: Remove before submission.
if __name__ == "__main__":    
    from data import load_data

    df_daylight1 = load_data("Data/Jonas1.json")
    df_daylight1["lighting_condition"] = "daylight"
    df_daylight2 = load_data("Data/Baisu1.json")
    df_daylight2["lighting_condition"] = "daylight"
    df_dark1 = load_data("Data/Tai4.json")
    df_dark1["lighting_condition"] = "dark"
    df_dark2 = load_data("Data/Zhi3.json")
    df_dark2["lighting_condition"] = "dark"

    df = pd.concat([df_daylight1, df_daylight2, df_dark1, df_dark2], ignore_index=True)
    print(df.columns)
    # print(df.head())
    # print(df.info())
    df[['correction_r4_R', 'correction_r4_G', 'correction_r4_B']] = df.apply(
        apply_correction,
        axis=1,
        r=4,
        correction_type='scaling',
        color_space='rgb'
    )
    print(df.head())
    print(df.info())


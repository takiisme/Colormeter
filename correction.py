import numpy as np
import pandas as pd

from constants import ColorSpace
from color_conversion import convert_rgb_cols


class CorrectionByScaling:
    def __init__(self, space: ColorSpace = ColorSpace.RGB, r: int = 4):
        self.space = space
        self.r = r
        
    def train(self, df: pd.DataFrame) -> None:
        # There is not training for scaling correction.
        pass
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling correction to the DataFrame measurements.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing measurement and reference columns.

        Returns: pd.DataFrame
            DataFrame with added correction columns.
        """
        channels = self.space.get_channels()
        meas_cols = [f'color_r{self.r}_{ch}' for ch in channels]
        ref_cols = [f'white_r{self.r}_{ch}' for ch in channels]
        assert all(col in df.columns for col in meas_cols + ref_cols), "DataFrame missing required measurement or reference columns."
        corr_cols = [f'correction_r{self.r}_{ch}' for ch in channels]
        
        if self.space == ColorSpace.RGB:
            scale_fn = self._scale_rgb_row
        else:
            scale_fn = self._scale_lab_row

        df[corr_cols] = df.apply(
            lambda row: scale_fn(
                row[meas_cols[0]], row[meas_cols[1]], row[meas_cols[2]],
                row[ref_cols[0]], row[ref_cols[1]], row[ref_cols[2]]
            ),
            axis=1,
            result_type='expand'
        )

        return df

    def _scale_rgb_row(
            self,
            meas_r: int, meas_g: int, meas_b: int,
            ref_r: int, ref_g: int, ref_b: int
        ) -> tuple[int, int, int]:
        """
        Apply scaling correction to a single RGB measurement.
        
        Parameters
        ----------
        meas_r, meas_g, meas_b : int
            Measured RGB values of the color.
        ref_r, ref_g, ref_b : int
            Measured RGB values of the white reference.

        Returns: tuple[int, int, int]
            Corrected RGB values.
        """
        scale_r = 255.0 / (ref_r if ref_r > 0 else 1.0)
        scale_g = 255.0 / (ref_g if ref_g > 0 else 1.0)
        scale_b = 255.0 / (ref_b if ref_b > 0 else 1.0)

        corr_r = np.clip(meas_r * scale_r, 0, 255).astype(int)
        corr_g = np.clip(meas_g * scale_g, 0, 255).astype(int)
        corr_b = np.clip(meas_b * scale_b, 0, 255).astype(int)

        return corr_r, corr_g, corr_b
    
    def _scale_lab_row(
            self,
            meas_l: float, meas_a: float, meas_b: float,
            ref_l: float, ref_a: float, ref_b: float
        ) -> tuple[float, float, float]:
        """
        Apply scaling correction to a single Lab measurement.

        Parameters
        ----------
        meas_l, meas_a, meas_b : float
            Measured Lab values of the color.
        ref_l, ref_a, ref_b : float
            Measured Lab values of the white reference.\

        Returns: tuple[float, float, float]
            Corrected Lab values.
        """
        scale_l = 100.0 / max(ref_l, 1e-6)
        corr_l = meas_l * scale_l
        corr_a = meas_a - ref_a
        corr_b = meas_b - ref_b

        # Clip to valid Lab range
        corr_l = np.clip(corr_l, 0.0, 100.0)
        corr_a = np.clip(corr_a, -128.0, 127.0)
        corr_b = np.clip(corr_b, -128.0, 127.0)

        return corr_l, corr_a, corr_b


class CorrectionByModel:
    def __init__(
            self,
            space: ColorSpace = ColorSpace.RGB,
            method: str = 'joint',
            pose: bool = True,
            degree: int = 1,
            reg_degree: float = 0.,
            reg_pose: float = 0.,
            boundary_penalty_factor: float = 1000.
        ):
        self.space = space
        self.method = method
        self.pose = pose
        self.degree = degree
        self.reg_degree = reg_degree
        self.reg_pose = reg_pose
        self.boundary_penalty_factor = boundary_penalty_factor
        
    def train(self, df: pd.DataFrame) -> None:
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

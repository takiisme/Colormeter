import numpy as np
import pandas as pd
from color_conversion import convert_rgb_cols
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
import math
from functools import partial
from scipy.optimize import minimize

from constants import ColorSpace
from color_conversion import convert_rgb_cols


class CorrectionByScaling:
    def __init__(self, space: ColorSpace = ColorSpace.RGB, r: int = 4):
        self.space = space
        self.r = r
        
    def train(self, df: pd.DataFrame) -> None:
        # There is no training for scaling correction.
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
            Measured Lab values of the white reference.

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
            boundary_penalty_factor: float = 1000,
            r: int = 4.
        ):
        self.space = space
        self.method = method
        self.pose = pose
        self.degree = degree
        self.reg_degree = reg_degree
        self.reg_pose = reg_pose
        self.boundary_penalty_factor = boundary_penalty_factor
        self.r = r
        self.coeffs = []
        self.m0 = None
        self.m1 = None
        self.m2 = None
        self.gt0 = None
        self.gt1 = None
        self.gt2 = None
    
    def build_design_matrix(self):
        """
        Build design matrix for polynomial optimization.

        joint:
            [1,
             degree = 1 terms,
             degree = 2 terms,
             ...
             degree = D terms]
            where terms are m0^i * m1^j * m2^k with i+j+k = d

        individual:
            For each channel:
            [1, m, m^2, ..., m^degree]

        If pose == True:
            Append pitch and roll at the end.
        """
        m0 = self.m0
        m1 = self.m1
        m2 = self.m2
        N = len(m0)

        # pose terms
        pose_terms = []
        if self.pose:
            pose_terms = [
                self.pitch.reshape(N, 1),
                self.roll.reshape(N, 1)
            ]
        
        if self.method == 'joint':
            terms = []

            # degree 0
            terms.append(np.ones((N, 1)))

            # degree 1 to degree D
            for d in range(1, self.degree + 1):
                for i, j, k in itertools.product(range(d + 1), repeat=3):
                    if i + j + k == d:
                        term = (m0 ** i) * (m1 ** j) * (m2 ** k)
                        terms.append(term.reshape(N, 1))

            # append pose at the end
            if self.pose:
                terms.extend(pose_terms)

            return np.hstack(terms)

        if self.method == 'individual':
            X_list = []
    
            for m in [m0, m1, m2]:
                terms = [np.ones((N, 1))]

                for d in range(1, self.degree + 1):
                    terms.append((m ** d).reshape(N, 1))

                if self.pose:
                    terms.extend(pose_terms)

                X_list.append(np.hstack(terms))

            return X_list

        
    # ==========
    # Compute unclipped corrected values
    # ==========
    def compute_corrected_values(self, coeffs, X, channel=None):
        if self.method == 'joint':
            # reshape to (num_coeffs, 3)
            coeffs = coeffs.reshape(-1, 3)
            corrected = X @ coeffs
            # Return 3 channel values
            return corrected[:, 0], corrected[:, 1], corrected[:, 2]
        elif self.method == 'individual':
            corrected = X[channel] @ coeffs[channel]
            # Return specific channel value
            return corrected

    # ==========
    # The loss is composed of 4 parts: MSE loss + boundary penalty + reg_degree + reg_pose (if applicable)
    # For joint method, it computes the loss for all 3 channels together
    # For individual method, it computes the loss for a specific channel
    # ==========
    def calculate_loss(self, coeffs, channel=None):
        
        X = self.build_design_matrix()

        if self.method == 'joint':
            # Compute MSE loss for all 3 channels
            c0, c1, c2 = self.compute_corrected_values(coeffs, X)
            mse0 = mean_squared_error(self.gt0, c0)
            mse1 = mean_squared_error(self.gt1, c1)
            mse2 = mean_squared_error(self.gt2, c2)
            mse_loss = (mse0 + mse1 + mse2) / 3
            # Boundary penalty for all 3 channels
            if self.space == ColorSpace.RGB:
                boundary_penalty = (
                    np.mean((c0 < 0) | (c0 > 255)) +
                    np.mean((c1 < 0) | (c1 > 255)) +
                    np.mean((c2 < 0) | (c2 > 255))
                ) * self.boundary_penalty_factor
            elif self.space == ColorSpace.LAB:
                boundary_penalty = (
                    np.mean((c0 < 0) | (c0 > 100)) +
                    np.mean((c1 < -128) | (c1 > 127)) +
                    np.mean((c2 < -128) | (c2 > 127))
                ) * self.boundary_penalty_factor
            # Compute regularization penalty
            reg_penalty = 0
            idx = 1  # skip constant term
            # Compute regularization for degree terms
            for d in range(1, self.degree + 1):
                # number of monomials with total degree = d
                num_terms = math.comb(d + 2, 2)
                # L1 regularization
                reg_penalty += (self.reg_degree ** d) * np.sum(np.abs(X[:, idx:idx + num_terms]))
                idx += num_terms
            # Compute regularization for pose terms
            if self.pose:
                reg_penalty += self.reg_pose * np.sum(np.abs(X[:, -2:])) # pitch and roll
            return mse_loss + boundary_penalty + reg_penalty
        elif self.method == 'individual':
            c = self.compute_corrected_values(coeffs, X, channel)
            # Compute MSE loss for specific channel
            if channel == 0:
                mse_loss = mean_squared_error(self.gt0, c)
            elif channel == 1:
                mse_loss = mean_squared_error(self.gt1, c)
            elif channel == 2:
                mse_loss = mean_squared_error(self.gt2, c)
            # Boundary penalty for specific channel
            if self.space == ColorSpace.RGB:
                boundary_penalty = (np.mean((c < 0) | (c > 255))) * self.boundary_penalty_factor
            elif self.space == ColorSpace.LAB and channel == 0:
                boundary_penalty = (np.mean((c < 0) | (c > 100))) * self.boundary_penalty_factor
            elif self.space == ColorSpace.LAB and channel in [1, 2]:
                boundary_penalty = (np.mean((c < -128) | (c > 127))) * self.boundary_penalty_factor
            # Compute regularization penalty
            reg_penalty = 0
            idx = 1  # skip constant term
            # Compute regularization for degree terms
            for d in range(1, self.degree + 1):
                # L1 regularization
                reg_penalty += (self.reg_degree ** d) * np.sum(np.abs(X[idx: idx + 1]))
                idx += 1
            # Compute regularization for pose terms
            if self.pose:
                reg_penalty += self.reg_pose * np.sum(np.abs(X[-2: ])) # pitch and roll
            return mse_loss + boundary_penalty + reg_penalty
        

    # ==========
    # Train the correction model    
    # Store and return the coefficients
    # ==========
    def train(self, df: pd.DataFrame):
        # Transfer the measured RGB to lab color space if it's going to be optimized in lab space
        if self.space == ColorSpace.LAB:
            convert_rgb_cols(df, prefix='gt__', to=ColorSpace.LAB)
            convert_rgb_cols(df, prefix=f'color_r{self.r}_', to=ColorSpace.LAB)
            self.m0 = df[f'color_r{self.r}_l'].values
            self.m1 = df[f'color_r{self.r}_a'].values
            self.m2 = df[f'color_r{self.r}_b'].values
            self.gt0 = df['gt__l'].values
            self.gt1 = df['gt__a'].values
            self.gt2 = df['gt__b'].values
        else:
            self.m0 = df[f'color_r{self.r}_R'].values
            self.m1 = df[f'color_r{self.r}_G'].values
            self.m2 = df[f'color_r{self.r}_B'].values
            self.gt0 = df['gt__R'].values
            self.gt1 = df['gt__G'].values
            self.gt2 = df['gt__B'].values
        self.pitch = df['pitch'].values
        self.roll = df['roll'].values
        
        if self.method == 'joint':
            X = self.build_design_matrix()
            x0 = np.zeros((3, X.shape[1])).flatten()
            # Create a partial function to compute loss
            loss_function = partial(
                self.calculate_loss
            )
            optimal_joint_coeffs = minimize(
                loss_function,
                x0=x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            self.coeffs = optimal_joint_coeffs.x.reshape(-1, 3)
        elif self.method == 'individual':
            self.coeffs = [None, None, None]
            X_list = self.build_design_matrix()
            
            for channel in range(3):
                Xc = X_list[channel]
                x0 = np.zeros(Xc.shape[1])
                # Create a partial function to compute loss for specific channel
                loss_function = partial(
                    self.calculate_loss,
                    channel=channel
                )
                optimal_individual_coeffs = minimize(
                    loss_function,
                    x0=x0,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-8}
                )
                self.coeffs[channel] = optimal_individual_coeffs.x
        print("train finished, coeffs shape:", self.coeffs.shape)
        print("coeffs:", self.coeffs[:5])

        return self.coeffs
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("coeffs:", self.coeffs)
        # Convert to Lab if needed
        if self.space == ColorSpace.LAB:
            convert_rgb_cols(df, prefix='gt__', to=ColorSpace.LAB)
            convert_rgb_cols(df, prefix=f'color_r{self.r}_', to=ColorSpace.LAB)
            self.m0 = df[f'color_r{self.r}_l'].values
            self.m1 = df[f'color_r{self.r}_a'].values
            self.m2 = df[f'color_r{self.r}_b'].values
            self.gt0 = df['gt__l'].values
            self.gt1 = df['gt__a'].values
            self.gt2 = df['gt__b'].values
        else:
            self.m0 = df[f'color_r{self.r}_R'].values
            self.m1 = df[f'color_r{self.r}_G'].values
            self.m2 = df[f'color_r{self.r}_B'].values
            self.gt0 = df['gt__R'].values
            self.gt1 = df['gt__G'].values
            self.gt2 = df['gt__B'].values
        self.pitch = df['pitch'].values
        self.roll = df['roll'].values
        # Compute unclipped corrected values
        if self.method == 'joint':
            X = self.build_design_matrix()
            c0, c1, c2 = self.compute_corrected_values(self.coeffs, X)
        elif self.method == 'individual':
            X_list = self.build_design_matrix()
            c0 = self.compute_corrected_values(self.coeffs, X_list, channel=0)
            c1 = self.compute_corrected_values(self.coeffs, X_list, channel=1)
            c2 = self.compute_corrected_values(self.coeffs, X_list, channel=2)
        # Clip to valid range and add to DataFrame
        if self.space == ColorSpace.RGB:
            df[f'correction_r{self.r}_R'] = np.clip(c0, 0, 255).astype(int)
            df[f'correction_r{self.r}_G'] = np.clip(c1, 0, 255).astype(int)
            df[f'correction_r{self.r}_B'] = np.clip(c2, 0, 255).astype(int)
        elif self.space == ColorSpace.LAB:
            df[f'correction_r{self.r}_l'] = np.clip(c0, 0.0, 100.0)
            df[f'correction_r{self.r}_a'] = np.clip(c1, -128.0, 127.0)
            df[f'correction_r{self.r}_b'] = np.clip(c2, -128.0, 127.0)
        return df
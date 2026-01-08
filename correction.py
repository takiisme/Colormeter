from types import ColorSpace
import pandas as pd

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

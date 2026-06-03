import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-wise engineered features for the AI4I predictive maintenance dataset.

    These features are safe because they do not use target information
    or statistics from the whole dataset.
    """

    df = df.copy()

    df["Temp_diff [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Power [W]"] = (
        df["Torque [Nm]"]
        * df["Rotational speed [rpm]"]
        * 2
        * np.pi
        / 60
    )

    df["Torque_Wear"] = (
        df["Torque [Nm]"] * df["Tool wear [min]"]
    )

    return df
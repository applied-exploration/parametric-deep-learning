import numpy as np
import pandas as pd


def load_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("data/dataset.csv")
    return df["features"].to_numpy(), df["label"].to_numpy()

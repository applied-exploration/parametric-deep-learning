import numpy as np
import pandas as pd


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/generated/dataset2.csv")
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)
    return (
        train["features"].to_numpy(),
        train["label"].to_numpy(),
        test["features"].to_numpy(),
        test["label"].to_numpy(),
    )

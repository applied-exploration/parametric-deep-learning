import numpy as np
import pandas as pd


def load_data(
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/generated/{}.csv".format(dataset_name))
    train = df.sample(frac=0.8, random_state=200)
    validation = train.sample(frac=0.3, random_state=200)
    train = train.drop(validation.index)
    test = df.drop(train.index)

    return (
        train["features"].to_numpy(),
        train["label"].to_numpy(),
        validation["features"].to_numpy(),
        validation["label"].to_numpy(),
        test["features"].to_numpy(),
        test["label"].to_numpy(),
    )

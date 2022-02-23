from data.generator import generator
from data.types import DataConfig
from config import dataconfig


def generate_dataset(data_config: DataConfig, display_plot: bool, type: str) -> None:

    # if type == "dataset1":
    #     generate_dataset1(config=data_config, display_plot=display_plot)

    if type == "dataset2":
        generator(config=data_config, display_plot=display_plot)


if __name__ == "__main__":
    generate_dataset(data_config=dataconfig, display_plot=True, type="dataset2")

from data.generator import generator
from data.types import DataConfig
from config import dataconfig_1, dataconfig_2


def generate_dataset(display_plot: bool) -> None:

    # generator(name="generated/dataset1", config=dataconfig_1, display_plot=display_plot)
    generator(name="generated/dataset2", config=dataconfig_2, display_plot=display_plot)


if __name__ == "__main__":
    generate_dataset(display_plot=True)

from data.generator import generator
from data.types import DataConfig
from config import dataconfig_1, dataconfig_2, dataconfig_3, dataconfig_basic


def generate_dataset(display_plot: bool) -> None:

    generator(config=dataconfig_basic, display_plot=display_plot)


if __name__ == "__main__":
    generate_dataset(display_plot=True)

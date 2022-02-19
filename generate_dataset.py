from data.dataset1_generator import generate_dataset1
from data.dataset2_generator import generate_dataset2
from data.data_utils import DataConfig

# import logging
# from readable_log_formatter import ReadableFormatter

# log = logging.getLogger()
# log.setLevel(logging.INFO)
# hndl = logging.StreamHandler()
# hndl.setFormatter(ReadableFormatter())
# log.addHandler(hndl)


def generate_dataset(data_config: DataConfig, display_plot: bool, type: str) -> None:

    if type == "dataset1":
        generate_dataset1(config=data_config, display_plot=display_plot)

    if type == "dataset2":
        generate_dataset2(config=data_config, display_plot=False)


if __name__ == "__main__":
    dataconfig = DataConfig(
        canvas_size=100,
        dataset_size=5,
        min_radius=5,
        max_radius=20,
        num_sample_points=100,
        num_circles=2,
    )
    generate_dataset(data_config=dataconfig, display_plot=True, type="dataset2")

from data.types import DataConfig, Program
from .utils import display_both
from typing import Callable
from .render import render


def visualize(dataconfig: DataConfig) -> Callable:
    def __visualize(
        ground_truth_programs: list[Program], predicted_programs: list[Program]
    ):

        ground_truth_program = ground_truth_programs[0]
        predicted_program = predicted_programs[0]

        primitives, _ = render(predicted_program)

        display_both(
            ground_truth_program,
            primitives,
            dataconfig,
        )

    return __visualize

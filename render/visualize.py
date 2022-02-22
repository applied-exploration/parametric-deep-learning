from data.types import DataConfig, Program, Point
from .utils import display_both
from typing import Callable
from .render import render


def visualize(dataconfig: DataConfig) -> Callable:
    def __visualize(x: list[list[Point]], y: list[Program]):

        primitives, _ = render(y[0])

        display_both(
            x[0],
            primitives,
            dataconfig,
        )

    return __visualize

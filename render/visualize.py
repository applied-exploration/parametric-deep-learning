from data.types import DataConfig, Program, Point
from .utils import display_both
from typing import Callable
from .render import render
from data.utils import write_definition


def visualize(dataconfig: DataConfig) -> Callable:
    def __visualize(x: list[list[Point]], y: list[Program]):

        primitives, modifiers = render(y[0])
        labels_collapsed: str = write_definition(primitives, modifiers)

        display_both(
            [x[0]],
            [(primitives, labels_collapsed)],
            dataconfig,
        )

    return __visualize

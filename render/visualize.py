from data.types import DataConfig, Program, Point, Primitive
from .utils import display_both
from typing import Callable
from .render import render
from data.utils import write_definition


def visualize(dataconfig: DataConfig) -> Callable:
    def __visualize(x: list[list[Point]], y: list[Program]):
        interactive_test_size = 2

        rendered_primitives: list[tuple[Primitive, str]] = []
        for i in range(interactive_test_size):
            primitives, modifiers = render(y[i])
            labels_collapsed: str = write_definition(primitives, modifiers)
            rendered_primitives.append((primitives, labels_collapsed))

        display_both(
            x[:interactive_test_size], rendered_primitives, dataconfig, interactive=True
        )

    return __visualize

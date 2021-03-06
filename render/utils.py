import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button
from data.types import DataConfig, Primitives, Primitive, Point
import numpy as np
from typing import Union
import itertools
from PIL import Image, ImageDraw


def display_features(objects: list, config: DataConfig) -> None:
    plt.xlim(-config.canvas_size / 2, config.canvas_size)
    plt.ylim(-config.canvas_size / 2, config.canvas_size)
    plt.gca().set_aspect("equal")

    for object in objects:
        zipped = list(zip(*object))
        plt.scatter(zipped[0], zipped[1])
    plt.show()


def display_program(rendered_primitives: list, config: DataConfig) -> None:
    plt.xlim(0, config.canvas_size)
    plt.ylim(0, config.canvas_size)
    # plt.gca().set_aspect("equal")

    patches = []
    fig, ax = plt.subplots()

    for object in rendered_primitives:
        patches.append(mpatches.Circle((object.x, object.y), object.r))

    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(colors)
    ax.add_collection(collection)
    # ax.add_line(line)

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    plt.show()


def display_both(
    grids: list[list[int]],
    rendered_primitives: list[tuple[Primitive, str]],
    config: DataConfig,
    interactive: bool = False,
) -> None:
    fig, axes = plt.subplots(ncols=2)

    # programs_unzipped = list(map(list, zip(*rendered_primitives)))

    def axis_setup():
        for ax in axes:
            ax.set(adjustable="box", aspect="equal")
            ax.set_xlim(0, config.canvas_size)
            ax.set_ylim(0, config.canvas_size)

    ys_grids = itertools.cycle(grids)
    ys_primitives = itertools.cycle(rendered_primitives)

    def add_data():
        next_element = next(ys_primitives)
        primitives = next_element[0]
        program_str = next_element[1]

        grid = next(ys_grids)
        axes[0].imshow(grid * 100, interpolation="nearest")

        for primitive in primitives:
            axes[1].add_patch(primitive.render())

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        axes[1].text(
            -0.15,
            -0.25,
            program_str,
            transform=axes[1].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

    axis_setup()
    add_data()

    if interactive:

        class Index:
            ind = 0

            def next(self, event):
                axes[0].cla()
                axes[1].cla()
                axis_setup()
                add_data()
                plt.draw()
                # fig.canvas.draw()

        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        callback = Index()
        bnext = Button(axnext, "Next")
        bnext.on_clicked(callback.next)
        # bprev = Button(axprev, 'Previous')
        # bprev.on_clicked(callback.prev)

        # cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=True)

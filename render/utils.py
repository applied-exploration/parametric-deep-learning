import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from data.types import DataConfig, Primitives, Point
import numpy as np
from typing import Union
import itertools


def display_features(objects: list, config: DataConfig) -> None:
    plt.xlim(-config.canvas_size / 2, config.canvas_size)
    plt.ylim(-config.canvas_size / 2, config.canvas_size)
    plt.gca().set_aspect("equal")

    for object in objects:
        zipped = list(zip(*object))
        plt.scatter(zipped[0], zipped[1])
    plt.show()


def display_program(rendered_primitives: list, config: DataConfig) -> None:
    plt.xlim(-config.canvas_size / 2, config.canvas_size)
    plt.ylim(-config.canvas_size / 2, config.canvas_size)
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
    points: list[list[Point]],
    rendered_primitives: list[Primitives],
    config: DataConfig,
    interactive: bool = False,
) -> None:
    fig, axes = plt.subplots(ncols=2)

    # programs_unzipped = list(map(list, zip(*rendered_primitives)))

    def axis_setup():
        for ax in axes:
            ax.set(adjustable="box", aspect="equal")
            ax.set_xlim(-config.canvas_size / 2, config.canvas_size / 2)
            ax.set_ylim(-config.canvas_size / 2, config.canvas_size / 2)

    ys_points = itertools.cycle(points)
    ys_primitives = itertools.cycle(rendered_primitives)

    def add_data():
        next_element = next(ys_primitives)
        primitives = next_element[0]
        program_str = next_element[1]

        for i, point in enumerate(next(ys_points)):
            axes[0].scatter(point[0], point[1])

        for i, primitive in enumerate(primitives):
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

        def onclick(event):
            axes[0].cla()
            axes[1].cla()
            axis_setup()
            add_data()

            fig.canvas.draw()

        cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

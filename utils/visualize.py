from turtle import fillcolor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from data.types import DataConfig
import numpy as np


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
    objects: list[list[float]], rendered_primitives: list, config: DataConfig
) -> None:
    fig, axes = plt.subplots(ncols=2)
    for ax in axes:
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set(adjustable="box", aspect="equal")
        ax.set_xlim(-config.canvas_size / 2, config.canvas_size)
        ax.set_ylim(-config.canvas_size / 2, config.canvas_size)

    cmap = plt.cm.get_cmap(plt.cm.viridis, 143).colors

    for i, object in enumerate(objects):
        zipped = list(zip(*object))
        axes[0].scatter(zipped[0], zipped[1])

    for i, primitive_collection in enumerate(rendered_primitives):
        for primitive in primitive_collection:
            axes[1].add_patch(
                mpatches.Circle(
                    (primitive.x, primitive.y), primitive.r, edgecolor=cmap[i],
                    facecolor=(0.0, 0.0, 0.0, 0.0),
                ),
            )

    # fig, axs = plt.subplots(ncols=2)
    # plt.xlim(-config.canvas_size / 2, config.canvas_size)
    # plt.ylim(-config.canvas_size / 2, config.canvas_size)
    # plt.gca().set_aspect("equal")

    plt.show()

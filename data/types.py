from dataclasses import dataclass
import math
import random
from dataclasses import dataclass
import numpy as np
from abc import ABC
from typing import Type, Optional
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw


Point = tuple[float, float]


class Instruction(ABC):
    name: str

    def get_params(self) -> tuple[float, float, float, float]:
        raise NotImplementedError()

    def get_params_dict(self) -> dict:
        raise NotImplementedError()


class Primitive(Instruction):
    def get_grid(self, width: int, height: int) -> np.ndarray:
        raise NotImplementedError()

    def get_position(self) -> tuple[float, float]:
        raise NotImplementedError()


class Modifier(Instruction):
    def apply(self, primitives: list[Primitive]) -> list[Primitive]:
        raise NotImplementedError()


def _add_rotation(angle: float, x: float, y: float) -> tuple[float, float]:
    rotation_matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    point = np.array([x, y])

    multiplied = np.matmul(rotation_matrix, point)

    x = multiplied[0]
    y = multiplied[1]

    return x, y


@dataclass(frozen=True)
class Circle(Primitive):
    r: float
    x: float
    y: float
    angle: float
    name: str = "CIRCLE"

    def get_random_point(self) -> tuple[float, float]:
        angle = random.random() * math.pi * 2
        x = math.cos(angle) * self.r + self.x
        y = math.sin(angle) * self.r + self.y

        return (x, y)

    def get_grid(self, width: int, height: int) -> np.ndarray:
        im = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        half_size = self.r / 2
        draw.rectangle(
            (
                self.x - half_size,
                self.y - half_size,
                self.x + half_size,
                self.y + half_size,
            ),
            outline=(1, 1, 1),
            width=1,
        )
        im = im.rotate(self.angle, center=(self.x, self.y), expand=False)
        return np.array(im)[:, :, 0]

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.r, self.x, self.y, self.angle)

    def get_params_dict(self) -> dict:
        return {"r": self.r, "x": self.x, "y": self.y, "angle": self.angle}

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def render(
        self,
        color: tuple[float, float, float] = (
            random.random(),
            random.random(),
            random.random(),
        ),
    ):
        return mpatches.Circle(
            (self.x, self.y),
            self.r,
            edgecolor=color,
            facecolor=(0.0, 0.0, 0.0, 0.0),
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Circle):
            return False
        return (
            math.isclose(self.r, __o.r, abs_tol=0.1)
            and math.isclose(self.x, __o.x, abs_tol=0.1)
            and math.isclose(self.y, __o.y, abs_tol=0.1)
            and math.isclose(self.angle, __o.angle, abs_tol=0.5)
        )


@dataclass(frozen=True)
class Square(Primitive):
    size: float
    x: float
    y: float
    angle: float
    name: str = "SQUARE"

    def get_random_point(self) -> tuple[float, float]:
        coinflip = random.random()

        point_from_center = ()
        if coinflip > 0.5:
            point_from_center = (
                random.choice([-1, 1]) * self.size,
                random.uniform(-1, 1) * self.size,
            )
        else:
            point_from_center = (
                random.uniform(-1, 1) * self.size,
                random.choice([-1, 1]) * self.size,
            )

        x = point_from_center[0] + self.x
        y = point_from_center[1] + self.y
        x, y = _add_rotation(self.angle, x, y)

        return (x, y)

    def get_grid(self, width: int, height: int) -> np.ndarray:
        im = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        half_size = self.size / 2
        draw.rectangle(
            (
                self.x - half_size,
                self.y - half_size,
                self.x + half_size,
                self.y + half_size,
            ),
            outline=(1, 1, 1),
            width=1,
        )
        im = im.rotate(self.angle, center=(self.x, self.y), expand=False)
        return np.array(im)[:, :, 0]

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.size, self.x, self.y, self.angle)

    def get_params_dict(self) -> dict:
        return {"size": self.size, "x": self.x, "y": self.y, "angle": self.angle}

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def render(
        self,
        color: tuple[float, float, float] = (
            random.random(),
            random.random(),
            random.random(),
        ),
    ):
        points = [
            [self.x + self.size, self.y + self.size],
            [self.x + self.size, self.y - self.size],
            [self.x - self.size, self.y - self.size],
            [self.x - self.size, self.y + self.size],
        ]

        points = np.array([list(_add_rotation(self.angle, x, y)) for x, y in points])

        return mpatches.Polygon(
            points,
            closed=True,
            edgecolor=color,
            facecolor=(0.0, 0.0, 0.0, 0.0),
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Square):
            return False
        return (
            math.isclose(self.size, __o.size, abs_tol=0.1)
            and math.isclose(self.x, __o.x, abs_tol=0.1)
            and math.isclose(self.y, __o.y, abs_tol=0.1)
            and math.isclose(self.angle, __o.angle, abs_tol=0.5)
        )


@dataclass(frozen=True)
class Triangle(Primitive):
    size: float
    x: float
    y: float
    angle: float
    name: str = "TRIANGLE"

    def get_random_point(self) -> tuple[float, float]:
        choose_side = random.choice([0, 1, 1])

        random_x = random.uniform(-1, 1) * self.size  # * random.choice([-1, 1])
        corresponding_y = (self.size - abs(random_x)) * math.sqrt(3.0)

        corresponding_y *= choose_side

        x = random_x + self.x
        y = corresponding_y + self.y

        x, y = _add_rotation(self.angle, x, y)

        return (x, y)

    def get_grid(self, width: int, height: int) -> np.ndarray:
        im = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        half_size = self.size / 2
        draw.polygon(
            [
                (self.x - half_size, self.y - half_size),
                (self.x + half_size, self.y - half_size),
                (self.x, self.y + half_size),

            ],
            outline=(100, 100, 100),
            width=1,
        )
        im = im.rotate(self.angle, center=(self.x, self.y), expand=False)
        return np.array(im)[:, :, 0]

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.size, self.x, self.y, self.angle)

    def get_params_dict(self) -> dict:
        return {"size": self.size, "x": self.x, "y": self.y, "angle": self.angle}

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def render(
        self,
        color: tuple[float, float, float] = (
            random.random(),
            random.random(),
            random.random(),
        ),
    ):

        points = [
            [self.x + self.size, self.y],
            [self.x, self.y + (math.sqrt(3.0) * self.size)],
            [self.x - self.size, self.y],
        ]

        points = np.array([list(_add_rotation(self.angle, x, y)) for x, y in points])

        return mpatches.Polygon(
            points,
            closed=True,
            edgecolor=color,
            facecolor=(0.0, 0.0, 0.0, 0.0),
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Triangle):
            return False
        return (
            math.isclose(self.size, __o.size, abs_tol=0.1)
            and math.isclose(self.x, __o.x, abs_tol=0.1)
            and math.isclose(self.y, __o.y, abs_tol=0.1)
            and math.isclose(self.angle, __o.angle, abs_tol=0.5)
        )


@dataclass(frozen=True)
class Translate(Modifier):
    x: float
    y: float
    index: int
    name: str = "TRANSLATION"

    def apply(
        self,
        primitives: list[Primitive],
    ) -> list[Primitive]:
        assert self.index < len(primitives), "This primitve does not exist"

        obj = primitives[self.index]
        obj_pos = obj.get_position()
        x = obj_pos[0] + self.x
        y = obj_pos[1] + self.y

        new_params = obj.get_params_dict()
        new_params["x"] = x
        new_params["y"] = y
        new_obj = type(obj)(**new_params)  # type: ignore

        new_primitives = primitives.copy()
        new_primitives[self.index] = new_obj

        return new_primitives

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.index, 0.0)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Translate):
            return False
        return (
            self.index == __o.index
            and math.isclose(self.x, __o.x, abs_tol=0.1)
            and math.isclose(self.y, __o.y, abs_tol=0.1)
        )


@dataclass(frozen=True)
class Rotate(Modifier):
    angle: float
    index: int
    name: str = "ROTATION"

    def apply(
        self,
        primitives: list[Primitive],
    ) -> list[Primitive]:
        assert self.index < len(primitives), "This primitve does not exist"

        obj = primitives[self.index]
        x, y = obj.get_position()

        new_x = (math.cos(self.angle) * x) - (math.sin(self.angle) * y)
        new_y = math.sin(self.angle) * x + math.cos(self.angle) * y

        new_params = obj.get_params_dict()
        new_params["x"] = new_x
        new_params["y"] = new_y
        new_params["angle"] = self.angle + new_params["angle"]
        new_obj = type(obj)(**new_params)  # type: ignore

        new_primitives = primitives.copy()
        new_primitives[self.index] = new_obj

        return new_primitives

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.angle, self.index, 0.0, 0.0)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Rotate):
            return False
        return self.index == __o.index and math.isclose(
            self.angle, __o.angle, abs_tol=0.5
        )


@dataclass(frozen=True)
class Constraint(Modifier):
    x: Optional[float]
    y: Optional[float]
    indicies: tuple[int, int]
    name: str = "CONSTRAINT"

    def apply(
        self,
        primitives: list[Primitive],
    ) -> list[Primitive]:
        obj_a = primitives[self.indicies[0]]
        obj_b = primitives[self.indicies[1]]

        con_x, con_y = self.x, self.y

        difference_vector = np.array(obj_b.get_position()) - np.array(
            obj_a.get_position()
        )

        constraint_vector = np.array(
            [con_x if con_x is not None else 0.0, con_y if con_y is not None else 0.0]
        )
        move_vector = difference_vector - constraint_vector

        x, y = obj_b.get_position()

        if self.x is not None:
            x = x - move_vector[0]
        if self.y is not None:
            y = y - move_vector[1]

        new_obj_a = type(obj_a)(**obj_a.get_params_dict())  # type: ignore

        new_params = obj_b.get_params_dict()
        new_params["x"] = x
        new_params["y"] = y
        new_obj_b = type(obj_b)(**new_params)  # type: ignore

        new_primitives = primitives.copy()
        new_primitives[self.indicies[0]] = new_obj_a
        new_primitives[self.indicies[1]] = new_obj_b

        return new_primitives

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.indicies[0], self.indicies[1])

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Constraint):
            return False
        return (
            self.indicies == __o.indicies
            and math.isclose(self.x, __o.x, abs_tol=0.1)
            and math.isclose(self.y, __o.y, abs_tol=0.1)
        )


all_instructions = {
    Circle: 0,
    Square: 1,
    Triangle: 2,
    Translate: 3,
    Rotate: 4,
    Constraint: 5,
}

Primitives = list[Primitive]
Modifiers = list[Modifier]
Program = list[Instruction]


@dataclass
class DataConfig:
    canvas_size: int
    min_radius: int
    max_radius: int
    num_sample_points: int
    dataset_size: int
    num_primitives: int
    random_primitives: bool
    num_modifiers: int
    instruction_embedding_size: int
    primitive_types: list[Type[Primitive]]
    modifier_types: list[Type[Modifier]]
    instructions_map: dict
    name: str

from dataclasses import dataclass
import math
import random
from dataclasses import dataclass
import numpy as np
from abc import ABC


@dataclass
class DataConfig:
    canvas_size: int
    min_radius: int
    max_radius: int
    num_sample_points: int
    dataset_size: int
    num_circles: int
    instruction_embedding_size: int
    max_definition_len: int


Point = tuple[float, float]


class Instruction(ABC):
    def get_random_point(self) -> tuple[float, float]:
        raise NotImplementedError()

    def get_params(self) -> tuple[float, float, float, float]:
        raise NotImplementedError()


@dataclass(frozen=True)
class Circle(Instruction):
    r: float
    x: float
    y: float
    name: str = "CIRCLE"

    def get_random_point(self) -> tuple[float, float]:
        angle = random.random() * math.pi * 2
        x = math.cos(angle) * self.r + self.x
        y = math.sin(angle) * self.r + self.y

        return (x, y)

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.r, self.x, self.y, 0.0)

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Translate(Instruction):
    x: float
    y: float
    index: int
    name: str = "TRANSLATION"

    def apply(
        self,
        primitives: list[Circle],
    ) -> list[Circle]:
        assert self.index < len(primitives), "This primitve does not exist"

        obj = primitives[self.index]
        obj_pos = obj.get_position()
        x = obj_pos[0] + self.x
        y = obj_pos[1] + self.y

        new_obj = Circle(obj.r, x, y)

        new_primitives = primitives.copy()
        new_primitives[self.index] = new_obj

        return new_primitives

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.index, 0.0)


@dataclass(frozen=True)
class Constraint(Instruction):
    x: float
    y: float
    indicies: tuple[int, int]
    name: str = "CONSTRAINT"

    def apply(
        self,
        primitives: list[Circle],
    ) -> list[Circle]:
        obj_a = primitives[self.indicies[0]]
        obj_b = primitives[self.indicies[1]]

        constraint_vector = np.array([self.x, self.y])
        difference_vector = np.array(obj_b.get_position()) - np.array(
            obj_a.get_position()
        )
        move_vector = difference_vector - constraint_vector

        x = obj_b.x - move_vector[0]
        y = obj_b.y - move_vector[1]

        obj_b = Circle(obj_b.r, x, y)
        obj_a = Circle(obj_a.r, obj_a.get_position()[0], obj_a.get_position()[1])

        new_primitives = primitives.copy()
        new_primitives[self.indicies[0]] = obj_a
        new_primitives[self.indicies[1]] = obj_b

        return new_primitives

    def get_params(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.indicies[0], self.indicies[1])


all_instructions = {Circle: 0, Translate: 1, Constraint: 2}

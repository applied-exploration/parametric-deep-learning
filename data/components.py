import math
import random
from dataclasses import dataclass
import numpy as np
from data_utils import DataConfig


@dataclass(frozen=True)
class Circle:
    r: float = 0.0
    position: tuple[float, float] = (0.0, 0.0)


    def get_random_point(self) -> tuple[float, float]:
        angle = random.random() * math.pi * 2
        x = math.cos(angle) * self.r + self.position[0]
        y = math.sin(angle) * self.r + self.position[1]

        return (x, y)
    
    def get_params(self) -> tuple[float, float, float]:
        return (self.r, self.position[0], self.position[1])
    
    def get_position(self) -> tuple[float, float]:
        return (self.position[0], self.position[1])


@dataclass(frozen=True)
class Translate:
    translation_x: float = 0.0
    translation_y: float = 0.0

    def apply(
        self,
        primitives: list[Circle],
        index: int,
    ) -> list[Circle]:
        assert index < len(primitives), "This primitve does not exist"
        
        obj = primitives[index]
        obj_pos = obj.get_position()
        x  = obj_pos[0] + self.translation_x
        y =  obj_pos[1] + self.translation_y

        new_obj = Circle(obj.r, (x, y))
        
        new_primitives = primitives.copy()
        new_primitives[index] = new_obj
        
        return new_primitives


@dataclass(frozen=True)
class Constrain:
    x: float = 0.0
    y: float = 0.0

    def apply(
        self,
        primitives: list[Circle],
        indexes: tuple[int, int]
    ) -> tuple[Circle, Circle]:
        obj_a = primitives[indexes[0]]
        obj_b = primitives[indexes[1]]
        
        constrain_vector = np.array([self.x, self.y])
        difference_vector = np.array(obj_b.get_position()) - np.array(obj_a.get_position())
        move_vector = difference_vector - constrain_vector
        
        x = obj_b.position[0] + move_vector[0]
        y = obj_b.position[1] + move_vector[0]
        
        obj_b = Circle(obj_b.r, (x, y))
        obj_a = Circle(obj_a.r, obj_a.get_position())
        
        
        return obj_a, obj_b



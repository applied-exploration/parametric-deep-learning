import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0

    def get_random_point(self) -> tuple[float, float]:
        angle = random.random() * math.pi * 2
        x = math.cos(angle) * self.radius + self.x
        y = math.sin(angle) * self.radius + self.y

        return (x, y)


@dataclass(frozen=True)
class Translate:
    x: float = 0.0
    y: float = 0.0

    def apply(
        self,
        x: float,
        y: float,
        random_translation_x: float,
        random_translation_y: float,
    ) -> tuple[float, float]:
        x += random_translation_x
        y += random_translation_y

        return x, y

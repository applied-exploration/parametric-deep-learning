#%%
from dataclasses import dataclass
from typing import Union


@dataclass
class Circle:
    r: float

    def get_parameters(self) -> list[float]:
        return [self.r, 0.0]


@dataclass
class Translation:
    x: float
    y: float

    def get_parameters(self) -> list[float]:
        return [self.x, self.y]


Instruction = Union[Circle, Translation]
Point = tuple[float, float]

#%%
from dataclasses import dataclass
from typing import Union

@dataclass
class Circle:
    r: float


@dataclass
class Translation:
    x: float
    y: float


Instruction = Union[Circle, Translation]
Point = tuple[float, float]

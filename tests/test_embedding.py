import numpy as np
from config import dataconfig
from data.types import Circle, Translate, Constraint
from embedding import embed_instructions

instructions = [
    Circle(0.1, 1.2, 2.2),
    Translate(0.2, 0.3, index=3),
    Translate(0.4, 0.6, index=4),
    Constraint(x=25, y=0, indicies=(0, 1)),
]

def test_embeddings():
    embedded = embed_instructions(dataconfig, instructions)
    assert all([x == y for x, y in zip(embedded, instructions)])

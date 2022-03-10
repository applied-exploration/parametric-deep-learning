import numpy as np
from config import dataconfig_3
from data.types import Circle, Translate, Constraint
from embedding.program_discretized import DiscretizedProgramStaticEmbeddings

instructions = [
    Circle(5.0, 51.0, 14.0, 0.0),
    Circle(6.0, 51.0, 14.0, 0.0),
    Translate(10.0, 15.0, index=0),
    Translate(15.0, 25.0, index=0),
    Constraint(x=25.0, y=90.0, indicies=(0, 1)),
]

dataconfig = dataconfig_3
program_embedding = DiscretizedProgramStaticEmbeddings(dataconfig)


def test_discretized_embeddings():
    embedded = program_embedding.program_to_tensor(instructions).unsqueeze(dim=0)
    embedded_reversed = program_embedding.tensor_to_programs(embedded)[0]
    assert all([x == y for x, y in zip(embedded_reversed, instructions)])

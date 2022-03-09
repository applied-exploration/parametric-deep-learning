from config import dataconfig_3
from data.types import Circle, Translate, Constraint
from embedding.program_mixed import MixedProgramStaticEmbeddings

instructions = [
    Circle(5.0, 51.0, 14.0, 1.0),
    Circle(6.0, 51.0, 14.0, 1.0),
    Translate(10.0, 15.0, index=0),
    Translate(15.0, 25.0, index=1),
    Constraint(x=25.0, y=90.0, indicies=(0, 1)),
]


dataconfig = dataconfig_3
program_embedding = MixedProgramStaticEmbeddings(dataconfig)


def test_mixed_embeddings_identical():
    embedded_1 = program_embedding.program_to_tensor(instructions).unsqueeze(dim=0)
    embedded_2 = program_embedding.program_to_tensor(instructions).unsqueeze(dim=0)
    assert program_embedding.loss(embedded_1, embedded_2) < 4


def test_mixed_embeddings_slightly_different():
    instructions2 = [
        Circle(50.0, 51.0, 14.0, 1.0),
        Circle(6.0, 51.0, 14.0, 1.0),
        Translate(10.0, 15.0, index=0),
        Translate(15.0, 25.0, index=1),
        Constraint(x=25.0, y=90.0, indicies=(0, 1)),
    ]
    embedded_1 = program_embedding.program_to_tensor(instructions).unsqueeze(dim=0)
    embedded_2 = program_embedding.program_to_tensor(instructions2).unsqueeze(dim=0)
    assert program_embedding.loss(embedded_1, embedded_2) > 20

from config import dataconfig_3
from data.types import Circle, Translate, Constraint
from embedding import embed_instructions, from_embeddings_to_instructions
from loss.compare_embeddings import compare_embedded_instructions_loss

instructions = [
    Circle(5.0, 51.0, 14.0, 1.0),
    Circle(6.0, 51.0, 14.0, 1.0),
    Translate(10.0, 15.0, index=0),
    Translate(15.0, 25.0, index=1),
    Constraint(x=25.0, y=90.0, indicies=(0, 1)),
]


dataconfig = dataconfig_3


def test_embeddings_identical():
    embedded_1 = embed_instructions(dataconfig)(instructions).unsqueeze(dim=0)
    embedded_2 = embed_instructions(dataconfig)(instructions).unsqueeze(dim=0)
    assert compare_embedded_instructions_loss(dataconfig)(embedded_1, embedded_2) < 4


def test_embeddings_slightly_different():
    instructions2 = [
        Circle(50.0, 51.0, 14.0, 1.0),
        Circle(6.0, 51.0, 14.0, 1.0),
        Translate(10.0, 15.0, index=0),
        Translate(15.0, 25.0, index=1),
        Constraint(x=25.0, y=90.0, indicies=(0, 1)),
    ]
    embedded_1 = embed_instructions(dataconfig)(instructions).unsqueeze(dim=0)
    embedded_2 = embed_instructions(dataconfig)(instructions2).unsqueeze(dim=0)
    assert compare_embedded_instructions_loss(dataconfig)(embedded_1, embedded_2) > 20

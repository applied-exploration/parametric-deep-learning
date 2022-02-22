from data.types import DataConfig

dataconfig = DataConfig(
    canvas_size=100,
    dataset_size=3,
    min_radius=5,
    max_radius=20,
    num_sample_points=100,
    num_circles=2,
    instruction_embedding_size=26,
    max_definition_len=10,  # maximum length of the program - we need this to know how many objects can be referenced in constraints
)

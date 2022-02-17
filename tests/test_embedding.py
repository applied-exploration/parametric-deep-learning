from utils.embedding import normalize_to_range, normalize_to_unit_vector
import numpy as np

input_values = [0.0, 12.0, 24.0, 48.0]
baseline_values = [0.0, 0.25, 0.5, 1.0]


def test_point_normalization() -> None:
    normalized_values = normalize_to_range(input_values)
    difference = np.sum(np.subtract(normalized_values, baseline_values))
    assert (
        -0.0001 <= difference <= 0.0001
    ), "Normalized values should be the same, it was {} rather than {}".format(
        normalized_values, baseline_values
    )

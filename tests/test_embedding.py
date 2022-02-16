import unittest

import sys
sys.path.append('.')

from utils.embedding import normalize_to_range, normalize_to_unit_vector
from parse import parse_program, parse_points
import numpy as np

input_values = [0., 12., 24., 48.]
baseline_values = [0., 0.25, 0.5, 1.0]

class TestEmbeddingMethods(unittest.TestCase):
    def test_point_normalization(self) -> None:
        print("Testing normalization.")
        normalized_values = normalize_to_range(input_values)
        difference = np.sum(np.subtract(normalized_values, baseline_values))
        assert -0.0001 <= difference <= 0.0001, "Normalized values should be the same, it was {}".format(normalized_values)
    
if __name__ == '__main__':
    unittest.main()
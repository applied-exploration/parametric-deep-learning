import unittest

import sys
sys.path.append('.')

from utils.loss import calculate_label_loss, calculate_symbolic_loss
from parse import parse_program, parse_points

example_program_good = "CIRCLE 1.3790708713968214\nTRANSLATION -24.86442070352715 8.947517177527358"
example_program_bad = "CIRCLE 0.3790708713968214\nTRANSLATION -22.86442070352715 7.947517177527358"
example_points = "-23.58597566123036 9.464638770735332\n-25.7535912561511 10.001657678922502\n-26.073447446210174 8.284123777038282\n-26.24211725720322 8.885964969408832\n-25.68404562582205 10.05659402567101\n-24.440242855542515 7.635301638214432\n-26.242491152563147 9.000036753200751\n-23.72819180435675 9.729067148434668\n-23.82385654821556 9.852538009893037\n-25.21018014795532 10.282540351455028"
parsed_program = parse_program(example_program_good)
estimated_program_values = [parsed_program[0].r, parsed_program[1].x, parsed_program[1].y]
actual_values = [1.3790708713968214, -24.86442070352715, 8.947517177527358]

class TestLossMethods(unittest.TestCase):
    def test_symbolic_loss(self) -> None:
        print("Testing loss based on distance between circle and points")
        symbolic_loss = calculate_symbolic_loss(example_program_good, example_points)
        assert -0.0001 <= symbolic_loss <= 0.0001, "Symbolic loss should be 0.0, it was {}".format(symbolic_loss)
    
    def test_label_loss(self) -> None:
        print("Testing loss based on difference between estimated and actual values")
        label_loss = calculate_label_loss(estimated_program_values, actual_values)
        assert -0.0001 <= label_loss <= 0.0001, "Label loss should be 0.0, it was {}".format(label_loss)
        
if __name__ == '__main__':
    unittest.main()
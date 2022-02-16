# import torch
from parse import parse_program, parse_points
import numpy as np
import math
from typing import Tuple, List

def calculate_label_loss(estimated_program_values:list, actual_values:list)->float:
    return np.sum(np.square(np.subtract(estimated_program_values, actual_values)))


def calculate_symbolic_loss(program:str, input_points:str)->float:
    est_program:list = parse_program(program)
    parsed_input_points:list[tuple] = parse_points(input_points)
    x, y, r = run_program(est_program)
    
    loss = 0
    
    for point in parsed_input_points:
        distance = math.hypot(point[0] - x, point[1] - y)
        loss += (distance - r) ** 2
        
    return loss
           
    
def run_program(program:list)->Tuple[float, float, float]:
    r = program[0].r
    x, y = program[1].x, program[1].y
    
    return x, y, r


def run_test()-> None:
    example_program = "CIRCLE 1.3790708713968214\nTRANSLATION -24.86442070352715 8.947517177527358"
    example_points = "-23.58597566123036 9.464638770735332\n-25.7535912561511 10.001657678922502\n-26.073447446210174 8.284123777038282\n-26.24211725720322 8.885964969408832\n-25.68404562582205 10.05659402567101\n-24.440242855542515 7.635301638214432\n-26.242491152563147 9.000036753200751\n-23.72819180435675 9.729067148434668\n-23.82385654821556 9.852538009893037\n-25.21018014795532 10.282540351455028"
    parsed_program = parse_program(example_program)
    estimated_program_values = [parsed_program[0].r, parsed_program[1].x, parsed_program[1].y]
    actual_values = [1.3790708713968214, -24.86442070352715, 8.947517177527358]
    
    print("Testing loss based on distance between circle and points")
    print(calculate_symbolic_loss(example_program, example_points))

    print("Testing loss based on difference between estimated and actual values")
    print(calculate_label_loss(estimated_program_values, actual_values))
    
if __name__ == "__main__":
    run_test()
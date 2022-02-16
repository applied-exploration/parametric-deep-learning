# import torch
from parse import parse_program, parse_points
import numpy as np

def calculate_label_loss(estimated_program_values, actual_values):
    return (estimated_program_values - actual_values) ** 2


def calculate_symbolic_loss(est_values, input_points):
    definition = "CIRCLE {}\nTRANSLATION {} {}".format(est_values[0], est_values[1], est_values[2])
    est_program = parse_program(definition)
    x, y, r = run_program(est_program)
    
    loss = 0
    
    for point in input_points:
        x = point[0] - x
        y = point[1] - y
        vector = np.array([x, y])
        mag = np.sqrt(vector.dot(vector))
        
        loss += (mag - r) ** 2
        
    return loss
           
    
def run_program(program):
    r = program[0].r
    x, y = program[1].x, program[1].y
    
    return x, y, r
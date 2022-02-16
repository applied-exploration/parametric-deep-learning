import numpy as np

def normalize_to_unit_vector(points:list[float])->list[float]:
    x = np.array(points)
    x = x/np.sqrt(x.dot(x))
    
    return x

def normalize_to_range(points:list[float])->np.ndarray:
    x = np.array(points)
    x = np.subtract(x, np.min(x))
    x = np.divide(x, np.max(x)-np.min(x))
        
    return x
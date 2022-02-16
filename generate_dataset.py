import csv 
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
    

            
def random_circle_point(radius:float)->tuple[float, float]:
    angle = random.random()*math.pi*2
    x = math.cos(angle)*radius
    y = math.sin(angle)*radius
    return x, y

def translation(x:float, y:float, random_translation_x:float, random_translation_y:float)->tuple[float, float]:
    translated_x = x+random_translation_x
    translated_y = y+random_translation_y
    return translated_x, translated_y

def generate_dataset(canvas_size:int, max_radius:int, num_sample_points:int, num_circles:int, display_plot:bool=False):
    column_names = ["features", "label"]

    df = pd.DataFrame(columns=column_names)
    
    for i in range(num_circles):
        random_samples = []
        plain_samples = []
        
        random_circle_radius = min(max_radius, random.random()*max_radius)
        random_translation_x = random.uniform(-1,1) * (canvas_size/2-max_radius)
        random_translation_y = random.uniform(-1,1) * (canvas_size/2-max_radius)
        
        for j in range(num_sample_points):
            x, y = random_circle_point(random_circle_radius)
            x, y = translation(x, y, random_translation_x, random_translation_y)
            random_samples.extend([str(x) + " " + str(y) + '\n'])
            
            if display_plot: plain_samples.extend([(x, y)])
        
        features_collapsed = "".join(random_samples)
        labels_collapsed = "".join(["CIRCLE {}\n".format(random_circle_radius), "TRANSLATION {} {}".format(random_translation_x, random_translation_y)])
        one_data_point = pd.Series(dict(zip(column_names, [features_collapsed, labels_collapsed])))
        df = df.append(one_data_point, ignore_index=True)
        
        ## Plot for reality check
        if display_plot:
            zipped = list(zip(*plain_samples))
            plt.gca().set_aspect('equal')
            plt.scatter(zipped[0], zipped[1])
            plt.xlim(-canvas_size/2,canvas_size)
            plt.ylim(-canvas_size/2,canvas_size)
            plt.show()
        
    df.to_csv('data/dataset.csv', index=True)

if __name__ == '__main__':
    canvas_size:int = 100
    max_radius:int = 20
    num_sample_points:int = 100
    num_circles:int = 500

    generate_dataset(canvas_size, max_radius, num_sample_points, num_circles, display_plot=False)   
        

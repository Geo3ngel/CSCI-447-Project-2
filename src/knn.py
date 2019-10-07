""" -------------------------------------------------------------
@file        knn.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       File that stores implementation of k-nearest neighbors, 
             edited k-nn and condensed k-nn
@TODO        Consider turning this into a class that stores k, training data, class_idx as variables
             since we are using them in multiple different functions
"""
import math
from collections import Counter
"""
@param class_cols   the indices of the classifier columns for these points
@return             the euclidean distance between two points
"""
def euc_distance(p1, p2, class_cols):
    dist = 0
    for idx in class_cols:
        dist += (float(p1[idx]) - float(p2[idx]))**2
    
    return math.sqrt(dist)

# Auxiliary function for sorting
def take_second(el):
    return el[1]

# Auxiliary function to get most common class in set of neighbors
def get_max_class(neighbors):
    classes = [el[0] for el in neighbors]
    return Counter(classes).most_common(1)[0][0]

# Auxiliary function to get mean class for regression
# TODO: Implement regression function from lecture(?)
def get_avg_class(neighbors):
    classes = [el[0] for el in neighbors]
    return sum(classes) / len(classes)

'''
@param x    datapoint we want to find nearest point to
@param z    set we are looking for the nearest point in
@brief      find the nearest point to x in the set z that has a different class than x
'''
def find_nearest(x, z, class_cols, class_idx):
    min_dist = float("inf")
    min_point = None
    for point in z:
        if point[class_idx] == x[class_idx]:
            continue
        dist = euc_distance(x, point, class_cols)
        if dist < min_dist:
            min_dist = dist
            min_point = point
    return min_point

# Calculate perforance
# I think we will most likely change this function
# Just using it for now to test edited_knn
def get_performance(k, type, data, class_idx, class_cols):
    correct_sum = 0
    for point in data:
        actual_class = point[class_idx]
        predicted_class = k_nearest_neighbors(k, type, data, point, class_idx, class_cols)
        if predicted_class == actual_class:
            correct_sum += 1
    
    return correct_sum / len(data)

    

'''
@param  k               The number of neighbors to find
@param  type            classification or regression
@param  training_data   Our training data set
@param  test_point      Point from test data
@param  class_idx       Index of the classifying attribute for this dataset
@return                 The predicted class
'''

def k_nearest_neighbors(k, type, training_data, test_point, class_idx, class_cols):
    distances = []
    for point in training_data:
        distances.append((point[class_idx], euc_distance(point, test_point, class_cols)))
    distances = sorted(distances, key=take_second)
    neighbors = distances[0:k]
    if type == 'classification':
        return get_max_class(neighbors)
    else:
        return get_avg_class(neighbors)

from copy import deepcopy
def edited_knn(k, type, training_data, class_idx, class_cols):
    edited_data = deepcopy(training_data)
    performance_improving = True
    current_performance = 0
    while performance_improving:
        for point in edited_data:
            correct_class = point[class_idx]
            predicted_class = k_nearest_neighbors(k, type, edited_data, point, class_idx, class_cols)

            if predicted_class != correct_class:
                edited_data.remove(point)
            
            past_performance = current_performance
            current_performance = get_performance(k, type, edited_data, class_idx, class_cols)
            if current_performance < past_performance:
                performance_improving = False
                break


def condensed_nn(training_data, class_idx, class_cols):
    z = []
    # Add the first point from the training data to z
    z.append(training_data[0])
    # Then remove it from the training data
    training_data[0].append('R')
    for x in training_data:
        if (x[-1] == 'R'): # Skip if point has been tagged for removal
            continue
        x_prime = find_nearest(x, z, class_cols, class_idx)
        if x_prime[class_idx] != x[class_idx]:
            z.append(x)
            # Tag x for removal
            x.append('R')
    return z


    
        
    
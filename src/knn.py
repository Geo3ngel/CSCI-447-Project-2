""" -------------------------------------------------------------
@file        knn.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       File that stores implementation of k-nearest neighbors, 
             edited k-nn and condensed k-nn
"""
import math
from collections import Counter
"""
@param p1           one of the points
@param p2           the other point
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
@param  k               The number of neighbors to find
@param  type            classification or regression
@param  training_data   Our training data set
@param  test_point      Point from test data
@param  class_idx       Index of the classifying attribute for this dataset
@return neighbors       array of k-nearest neighbors to test_point
'''
def k_nearest_neighbors(k, type, training_data, test_point, class_idx, class_cols):
    distances = []
    for point in training_data:
        distances.append((point[class_idx], euc_distance(point, test_point, class_cols)))
    distances = sorted(distances, key=take_second)
    neighbors = distances[0:k]
    if type == 'classification':
        return get_max_class(neighbors)
    elif type == 'regression':
        return get_avg_class(neighbors)
        
        



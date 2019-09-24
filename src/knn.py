""" -------------------------------------------------------------
@file        knn.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       File that stores implementation of k-nearest neighbors, 
             edited k-nn and condensed k-nn
"""
import math
'''
@param p1
@param p2
@param class_cols   the indices of the classifier columns for these points
'''
def euc_distance(p1, p2, class_cols):
    dist = 0
    for idx in class_cols:
        dist += float(p1[idx])**2 + float(p2[idx])**2
    
    return math.sqrt(dist)



'''
@param  k               The number of neighbors to find
@param  training_data   Our training data set
@param  test_point      Point from test data
@param  class_idx       Index of the classifying attribute for this dataset
@return neighbors       array of k-nearest neighbors to test_point
'''
def get_nearest_neighbors(k, training_data, test_point, class_idx, class_cols):
    distances = []
    for point in training_data:
        distances.append([point[class_idx], euc_distance(point, test_point, class_cols)])
    
    print("TEST POINT: ", test_point)
    print("DISTANCES: ")
    for d in distances:
        print(d)
        
        



""" -------------------------------------------------------------
@file        k_nn.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       File that stores implementation of k-nearest neighbors, 
             edited k-nn and condensed k-nn
"""

'''
@param  k               The number of neighbors to find
@param  training_data   Our training data set
@param  test_point      Point from test data we are predicting the classification of
@return neighbors       array of k-nearest neighbors to test_point
'''
def get_nearest_neighbors(k, training_data, test_point, class_col):
    distances = []
    for point in training_data:
        print(point)
        



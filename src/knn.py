""" -------------------------------------------------------------
@file        knn.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       File that stores implementation of k-nearest neighbors, 
             edited k-nn and condensed k-nn
             since we are using them in multiple different functions
"""
import math
from collections import Counter
from copy import deepcopy
import operator
import random

class knn:
    def __init__(self, k, type, class_idx, class_cols):
        self.k = k
        self.type = type
        self.class_idx = class_idx
        self.class_cols = class_cols
    
    def set_k(self, k):
        self.k = k
    
    def get_k(self):
        return self.k

    def set_type(self, type):
        self.type = type
    
    def get_type(self):
        return self.type
    



    """
    @param class_cols   the indices of the classifier columns for these points
    @return             the euclidean distance between two points
    """

    # Computes the euclidean distances for any dimension, so long as the data instances are consistent in dimension.
    def euclidean_distance(self, data_instance_a, data_instance_b):
        distance = 0
        for x in range(len(data_instance_a)):
            distance += pow((data_instance_a[x] - data_instance_b[x]), 2)
        return math.sqrt(distance)
    
    # Computes euclidean distance using only the classifier cols
    def euc_distance(self, point_a, point_b):
        distance = 0
        for x in self.class_cols:
            distance += pow((float(point_a[x]) - float(point_b[x])), 2)
        return math.sqrt(distance)
    
    # Auxiliary function for sorting
    def take_second(self, el):
        return el[1]

    # Auxiliary function to get most common class in set of neighbors
    # @param neighbors  set of tuples in form (class, distance)
    def get_max_class(self, neighbors):
        classes = [el[0] for el in neighbors]
        return Counter(classes).most_common(1)[0][0]

    # Auxiliary function to get mean class for regression
    # TODO: Implement regression function from lecture(?)
    def get_avg_class(self, neighbors):
        classes = [float(el[0]) for el in neighbors]
        return sum(classes) / len(classes)

    '''
    @param x    datapoint we want to find nearest point to
    @param z    set we are looking for the nearest point in
    @brief      find the nearest point to x in the set z that has a different class than x
    '''
    def find_nearest(self, x, z):
        min_dist = float("inf")
        min_point = None
        for point in z:
            # if point[self.class_idx] == x[self.class_idx]:
            #     continue
            dist = self.euc_distance(x, point)
            if dist < min_dist:
                min_dist = dist
                min_point = point
        return min_point

    # Calculate perforance
    # I think we will most likely change this function
    # Just using it for now to test edited_knn
    def get_performance(self, training_data, validation_data):
        correct_sum = 0
        for point in validation_data:
            actual_class = point[self.class_idx]
            predicted_class = self.k_nearest_neighbors(training_data, point)
            if predicted_class == actual_class:
                correct_sum += 1
        
        return correct_sum / len(validation_data)

        

    '''
    @param  k               The number of neighbors to find
    @param  type            classification or regression
    @param  training_data   Our training data set
    @param  test_point      Point from test data
    @param  class_idx       Index of the classifying attribute for this dataset
    @return                 The predicted class
    '''

    def get_k_nearest_neighbors(self, training, point, k_nearest):
        # Calculates the distance from the point to all other points int he training set.
        distances = []
        for iter in range(len(training)):
            dist=self.euclidean_distance(point, training[iter])
            distances.append((training[iter], dist))
        distances.sort(key=operator.itemgetter(1))
        
        # Collects a list of k points with the smallest distance to point.
        neighbors = []
        for iter in range(k_nearest):
            neighbors.append(distances[iter][0])
        return neighbors
    
    def k_nearest_neighbors(self, training_data, test_point):
        # print("TRAINING DATA:")
        # for row in training_data:
        #     print(row)
        # print('------------------------------')

        distances = []
        for point in training_data:
            distances.append((point[self.class_idx], self.euc_distance(point, test_point)))
        distances = sorted(distances, key=self.take_second)
        neighbors = distances[0:self.k]
        if self.type == 'classification':
            return self.get_max_class(neighbors)
        else:
            return self.get_avg_class(neighbors)

    # Handles voting of each k nearest neighbor to classify point.
    def majority_vote(self, neighbors):
        classes = {}
        for iter in range(len(neighbors)):
            response = neighbors[iter][-1]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1
        sorted_votes = sorted(classes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]
    
    def edited_knn(self, training_data, validation_data):
        edited_data = deepcopy(training_data)
        performance_improving = True
        current_performance = 0
        while performance_improving:
            for point in edited_data:
                correct_class = point[self.class_idx]
                predicted_class = self.k_nearest_neighbors(edited_data, point)
                # print('CORRECT CLASS: ', correct_class)
                # print("PREDICTED CLASS: ", predicted_class)
                if predicted_class != correct_class:
                    # print('REMOVING POINT')
                    edited_data.remove(point)
            # END FOR LOOP
            #Check performance at end of each scan
            past_performance = current_performance
            current_performance = self.get_performance(edited_data, validation_data)
            print("PAST PERFORMANCE:    ", past_performance)
            print("CURRENT PERFORMANCE: ", current_performance)
            print('------------------------------------')
            if current_performance < past_performance:
                performance_improving = False
        # END WHILE LOOP
        
        return edited_data

    
    def condensed_nn(self, training_data):
        z = []
        # Add the first point from the training data to z
        rand_idx = random.randint(0, len(training_data)-1)
        z.append(training_data[rand_idx])
        # Then tag it for removal
        training_data[rand_idx].append('R')
        past_length = -1
        current_length = 0
        print("Z:",z)
        while past_length < current_length:
            for x in training_data:
                if (x[-1] == 'R'): # Skip if point has been tagged for removal
                    continue
                print("X: ", x)
                x_prime = self.find_nearest(x, z)
                print("X PRIME: ", x_prime)
                if x_prime[self.class_idx] != x[self.class_idx]:
                    z.append(x)
                    # Tag x for removal
                    x.append('R')
            past_length = current_length
            current_length = len(z)
            print('PAST LENGTH: ', past_length)
            print('CURRENT LENGTH: ', current_length)
        return z
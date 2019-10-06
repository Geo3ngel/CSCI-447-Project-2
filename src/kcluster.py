""" ---------------------------------------------------
@file       kcluster.py
@authors    George Engel, Troy Oster, Dana Parker, Henry Soule
@brief      Stores functionality for k-means clusters
            and k-medoids clustering
"""

# ---------------------------------------------------
# Imports

import numpy as np
import random as rand
#import matplotlib.pypolt as plt
import sys

# ---------------------------------------------------
# Our main class for kcluster.py
class kcluster:

    """ ---------------------------------------------------
    @param  in_db   The input database object to perform our functionality upon
    @brief    The constructor for kcluster
    """
    def __init__(self, in_num_clusters, in_max_iters, in_tolerance):
        self.db = self.temp_data('temp_data.txt')
        self.k = in_num_clusters
        self.max_iters = in_max_iters
        self.tol = in_tolerance
        self.centroids = []
        self.clusters = []

        self.calc_centroids()
        self.calc_clusters()

    
    """ ---------------------------------------------------
    @param  db   A list of examples (lists) that is our data
        TODO: eventually make db use self.db.get_data() instead
    
    @brief    Acquires the minimum and maximum value of each attribute
    """
    def find_attrs_min_max(self):
        attr_mins = [sys.maxsize for i in range(len(self.get_db()[0]))]
        attr_maxs = [-sys.maxsize for i in range(len(self.get_db()[0]))]

        # For each example in our data set...
        for curr_ex in self.get_db():
            
            # For each attribute in our current example...
            for attr_idx in range(len(curr_ex)):
                
                # If the current example's attribute value
                # at attr_idx is less than
                # the current minimum for that attribute...
                if (curr_ex[attr_idx] < attr_mins[attr_idx]):

                    # Then set the new minimum
                    attr_mins[attr_idx] = curr_ex[attr_idx]

                # If the current example's attribute value
                # at attr_idx is less than
                # the current minimum for that attribute...
                if (curr_ex[attr_idx] > attr_maxs[attr_idx]):

                    # Then set the new maximum
                    attr_maxs[attr_idx] = curr_ex[attr_idx]

        return attr_mins, attr_maxs

    """ ---------------------------------------------------
    @param  db   A list of examples (lists) that is our data
        TODO: eventually make db use self.db.get_data() instead

    @brief    Initializes a list of size
                (number of attributes) times k to some number
                within an attributes respective range of possible values
    """
    def init_centroids(self, attr_mins, attr_maxs):
        
        # Initialize our list of centroids to
        # A list of zeros of dimensions
        # (num of attrs) times k
        centroids = [[0 for i in range(len(self.db[0]))] \
            for j in range(self.get_k())]

        # For every centroid...
        for curr_cent in centroids:

            # For every attribute...
            for idx in range(len(curr_cent)):

                # Initialize the centroid's attribute value
                # to some number in the range
                # of possible attribute values
                curr_cent[idx] = rand.uniform( \
                    attr_mins[idx] + 1, \
                    attr_maxs[idx] - 1)

        self.set_centroids(centroids)

    """ ---------------------------------------------------
    @param  x   An example from our database (i.e. a list of attributes)
    @param  y   An example from our database (i.e. a list of attributes)

    @brief     Finds the euclidean distance between x and y
    @return    The Euclidean distance between x and y
    """
    def kmeans_euc_dist(self, x, y):
        
        # The sum of squared distances of examples x and y
        sum = 0

        # For each
        # (len(x) and len(y) are completely interchangable)
        for attr_idx in range(len(x)):
            sum += (x[attr_idx] - y[attr_idx])**2

        return sum

    def update_centroid(self, centroid, n, examples_in_clust):
        for idx in range(len(centroid)):
            centroid[idx] = \
                (centroid[idx] * (n-1) \
                + examples_in_clust[idx]) / float(n)
        
        return centroid
    
    def assign_to_clust(self, example):
        min_dist = sys.maxsize
        min_cent_idx = -1    # -1 being a default value

        # For each centroid...
        for idx in range(len(self.get_centroids())):

            # Find the distance from
            # current example to current centroid
            curr_dist = self.kmeans_euc_dist(self.get_centroids()[idx], example)

            # If that distance is less than
            # the current min distance...
            if (curr_dist < min_dist):

                # Set the new minimum distance
                min_dist = curr_dist

                # Det the new minimum distant centroid index
                min_cent_idx = idx
        
        return min_cent_idx
    
    def calc_centroids(self):
        db = self.get_db()

        # Get the attribute minimum and maximum values
        attr_mins, attr_maxs = self.find_attrs_min_max()

        # Initialize the centroids randomly
        self.init_centroids(attr_mins, attr_maxs)
        centroids = self.get_centroids()

        # Holds the number of examples contained in each cluster
        size_of_clust = [0 for i in range(self.get_k())]

        # Holds which cluster each example belongs to
        ex_to_clust = [0 for i in range(len(db))]

        # Calculate the centroids
        for curr_iter in range(self.get_max_iters()):
            
            # We need a way to keep track of whether
            # the centroids after an iteration
            centroids_were_changed = False

            for curr_ex_idx in range(len(db)):

                # Find which cluster we are in
                cent_idx = self.assign_to_clust(db[curr_ex_idx])

                size_of_clust[cent_idx] += 1

                centroids[cent_idx] = self.update_centroid(\
                    centroids[cent_idx], \
                    size_of_clust[cent_idx], \
                    db[curr_ex_idx])
                
                if (cent_idx != ex_to_clust[curr_ex_idx]):
                    centroids_were_changed = True
                
                ex_to_clust[curr_ex_idx] = cent_idx

            if (centroids_were_changed is False):
                break

        self.set_centroids(centroids)

    def calc_clusters(self):
        
        # Initialize list of clusters
        clusters = [[] for i in range(self.get_k())]

        for ex in self.get_db():
            cent_idx = self.assign_to_clust(ex)
            clusters[cent_idx].append(ex)

        self.set_clusters(clusters)

    # ---------------------------------------------------
    # Setters

    def set_centroids(self, in_centroids):
        self.centroids = in_centroids

    def set_clusters(self, in_clusters):
        self.clusters = in_clusters

    # ---------------------------------------------------
    # Getters

    def get_k(self):
        return self.k

    def get_db(self):
        return self.db

    def get_max_iters(self):
        return self.max_iters
    
    def get_tol(self):
        return self.tol
    
    def get_centroids(self):
        return self.centroids
    
    def get_clusters(self):
        return self.clusters

    # ------------------------------------
    # TODO: remove when databases work
    # temp stuff until databases work
    def temp_data(self, fileName): 
  
        # Read the file, splitting by lines 
        f = open(fileName, 'r'); 
        lines = f.read().splitlines(); 
        f.close(); 
    
        items = []; 
    
        for i in range(1, len(lines)): 
            line = lines[i].split(','); 
            itemFeatures = []; 
    
            for j in range(len(line)-1): 
                v = float(line[j]); # Convert feature value to float 
                itemFeatures.append(v); # Add feature value to dict 
    
            items.append(itemFeatures); 
    
        rand.shuffle(items); 
    
        return items;
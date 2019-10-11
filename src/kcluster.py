""" ---------------------------------------------------
@file       kcluster.py
@authors    George Engel, Troy Oster, Dana Parker, Henry Soule
@brief      Stores functionality for k-means clusters
            and k-medoids clustering
"""

# ---------------------------------------------------
# Imports

#import numpy as np
import random as rand
from copy import deepcopy
#import matplotlib.pypolt as plt
import sys
import math

# ---------------------------------------------------
# Our main class for kcluster.py
class kcluster:

    """ ---------------------------------------------------
    @param  in_db   The input database object to perform our functionality upon
    @brief    The constructor for kcluster
    """
    def __init__(self, in_num_clusters, in_max_iters, in_db, class_cols, in_func):
        self.db = in_db
        self.k = int(in_num_clusters)
        self.max_iters = in_max_iters
        # self.tol = in_tolerance
        self.class_cols = class_cols
        self.centroids = []
        self.medoids = []
        self.kmeans_clusters = []

        if in_func == 'k-means':

            self.calc_centroids()
            print("Finished Centroids.")
            self.calc_kmeans_clusters()
            print("Finished kmeans clusters.")
        
        elif in_func == 'k-medoids':
        
            self.calc_medoids_and_clusters()
            print("Finished medoids and clusters.")


    
    """ ---------------------------------------------------
    @param  db   A list of examples (lists) that is our data
        TODO: eventually make db use self.db.get_data() instead
    
    @brief    Acquires the minimum and maximum value of each attribute
    """
    def find_attrs_min_max(self):
        attr_mins = [sys.maxsize for i in range(len(self.get_db()[0]))]
        attr_maxs = [-sys.maxsize for i in range(len(self.get_db()[0]))]
        discrete_attrs_vals = [[] for i in range(len(self.get_db()[0]))]

        # For each example in our data set...
        for curr_ex in self.get_db()[:]:
            
            # For each attribute in our current example...
            for attr_idx in range(len(curr_ex)):

                # If our current attribute is continuous...
                if (type(curr_ex[attr_idx]) != str):

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
                
                # If our current attribute is discrete...
                else:
                    discrete_attrs_vals[attr_idx].append(curr_ex[attr_idx])

        return attr_mins, attr_maxs, discrete_attrs_vals

    """ ---------------------------------------------------
    @param TODO: add params

    @brief    Initializes a list of size
                (number of attributes) times k to some number
                within an attributes respective range of possible values
    """
    def init_centroids(self, attr_mins, attr_maxs, discrete_attrs_vals):
        
        # Initialize our list of centroids to
        # a list of zeros of dimensions
        # (num of attrs) times k
        centroids = [[0 for i in range(len(self.get_db()[0]))] \
            for j in range(self.get_k())]

        # For every centroid...
        for curr_cent in centroids:

            # For every attribute...
            for idx in range(len(curr_cent)):

                # If the current attribute is continuous...
                if (len(discrete_attrs_vals[idx]) == 0):
                
                    # Initialize the centroid's attribute value
                    # to some number in the range
                    # of possible attribute values
                    curr_cent[idx] = rand.uniform( \
                        attr_mins[idx] + 1, \
                        attr_maxs[idx] - 1)
                
                # If the current attribute is discrete...
                else:
                    rand_idx = rand.randrange(len(discrete_attrs_vals[idx]))
                    curr_cent[idx] = discrete_attrs_vals[idx][rand_idx]
                    
        self.set_centroids(centroids)

    """ ---------------------------------------------------    
    TODO: add info 
    """
    def init_medoids(self):

        rv = []
        iter = 0

        while iter < self.get_k():
            
            # Generate a random index
            rand_num = rand.randint(0, len(self.get_db()) - 1)

            # If the random index isn't already in medoids
            # then add it to the medoids list
            if rand_num not in rv:
                rv.append(rand_num)
                iter += 1
            
        self.set_medoids(rv)

    """ ---------------------------------------------------
    @param  x   An example from our database (i.e. a list of attributes)
    @param  y   An example from our database (i.e. a list of attributes)

    @brief     Finds the euclidean distance between x and y
    @return    The Euclidean distance between x and y
    """
    def euc_dist(self, x, y):
        
        # The sum of squared distances of examples x and y
        sum = 0
        
        # For each
        # (len(x) and len(y) are completely interchangable)
        for attr_idx in self.class_cols:
            # If the current attribute is continuous...
            if len(x) < attr_idx+1:
                print("REF POINT: ", x)
            if (type(x[attr_idx]) != str):
                sum += (x[attr_idx] - y[attr_idx])**2
            
            # If the current attribute is discrete...
            else:

                # If the discrete attributes aren't the same...
                if (x[attr_idx] != y[attr_idx]):
                    sum += 1

        return math.sqrt(sum)

    # """ ---------------------------------------------------
    # TODO: add info
    # """
    # def update_centroid(self, centroid, n, examples_in_clust, discrete_attrs_vals):
    #     print(examples_in_clust)
    #     for attr_idx, attr in enumerate(centroid):
    #         centroid[attr_idx] = \
    #             (centroid[attr_idx] * (n-1) \
    #             + examples_in_clust[attr_idx]) / float(n)
        
    #     return centroid
    
    """ ---------------------------------------------------
    TODO: add info
    """
    def assign_ex(self, ex, using_centroids, skip_empty = False):
        min_dist = sys.maxsize
        min_idx = -1    # -1 being a default value
        ref_points = []

        if using_centroids is True:
            ref_points = self.get_centroids()[:]
        else:
            for med_idx in self.get_medoids()[:]:
                ref_points.append(self.get_db()[:][med_idx])

        # For each centroid...
        for idx in range(len(ref_points)):
            # Find the distance from
            # current example to current centroid
            if len(ref_points[idx]) > 0:
                curr_dist = self.euc_dist(ref_points[idx], ex)
                # If that distance is less than
                # the current min distance...
                if (curr_dist < min_dist):

                    # Set the new minimum distance
                    min_dist = curr_dist

                    # Set the new minimum distant centroid index
                    min_idx = idx
        
        return min_idx
    
    """ ---------------------------------------------------
    TODO: add info
    """
    def calc_centroids(self):
        db = self.get_db()[:]

        # Get the attribute minimum and maximum values
        attr_mins, attr_maxs, discrete_attrs_vals = self.find_attrs_min_max()

        # Initialize the centroids randomly
        clusters_arent_empty = False
        loop_count = 0
        # while clusters_arent_empty is False:
        self.init_centroids(attr_mins, attr_maxs, discrete_attrs_vals)
        print("Inited centroids")
        # clusters = [[] for i in range(len(self.get_centroids()[:]))]

            # for ex_idx, ex in enumerate(self.get_db()[:]):
            #     # Find which cluster we are in
            #     cent_idx = self.assign_ex(ex, True)
            #     clusters[cent_idx].append(ex)

            # clusters_arent_empty = True
            # for c in clusters:
            #     if len(c) == 0:
            #         clusters_arent_empty = False
            # loop_count += 1
            # if loop_count > 10:
            #     self.fix_empty_clusters(clusters)
            

        # Holds the number of examples contained in each cluster
        # size_of_clust = [0 for i in range(self.get_k())]

        # Calculate the centroids
        print("Max iters: ", self.get_max_iters())
        for curr_iter in range(self.get_max_iters()):
            print("Current iter: ", curr_iter)

            # Holds which cluster each example belongs to
            ex_to_clust = [-1 for i in range(len(self.get_db()))]

            # Creates a list of examples per cluster
            clusters = [[] for i in range(self.get_k())]

            centroids = self.get_centroids()[:]

            for ex_idx, ex in enumerate(self.get_db()[:]):

                # Find which cluster we are in
                cent_idx = self.assign_ex(ex, True)

                # size_of_clust[cent_idx] += 1

                # centroids[cent_idx] = self.update_centroid( \
                #     centroids[cent_idx], \
                #     size_of_clust[cent_idx], \
                #     db[curr_ex_idx], \
                #     discrete_attrs_vals)
                
                # if (cent_idx != ex_to_clust[curr_ex_idx]):
                #     centroids_were_changed = True
                
                ex_to_clust[ex_idx] = cent_idx

                clusters[cent_idx].append(ex)

            new_centroids = [[] for i in range(len(centroids))]

            # For every centroid...
            for cent_idx in range(len(centroids)):

                if len(clusters[cent_idx]) != 0:

                    attr_avgs = [0 if len(discrete_attrs_vals[i]) == 0 \
                        else {} \
                        for i in range(len(centroids[cent_idx]))]

                    # For every example in that centroid's respective cluster
                    for ex_idx, ex in enumerate(clusters[cent_idx]):

                        # For every attribute in that example...
                        for attr_idx, attr in enumerate(ex):

                            # If the current attribute is quantitative...
                            if len(discrete_attrs_vals[attr_idx]) == 0:
                                attr_avgs[attr_idx] += attr
                            
                            # If the current attribute is categorical...
                            else:
                                if attr in attr_avgs[attr_idx]:
                                    attr_avgs[attr_idx][attr] += 1
                                else:
                                    attr_avgs[attr_idx][attr] = 1

                    # For every attribute
                    for attr_idx, attr in enumerate(attr_avgs):
                        
                        if type(attr) == dict:
                            max_val = max(attr_avgs[attr_idx].values())
                            max_key = ''
                            for key, val in attr_avgs[attr_idx].items():
                                if val == max_val:
                                    max_key = key
                            attr_avgs[attr_idx] = max_key

                        else:
                            if len(clusters[cent_idx]) != 0:
                                attr_avgs[attr_idx] /= len(clusters[cent_idx])

                    new_centroids[cent_idx] = attr_avgs[:]

            # if (centroids_were_changed is False):
            #     break

            self.set_centroids(new_centroids)
        print('Finished Iterations')

    """ ---------------------------------------------------
    TODO: add info
    """
    def calc_medoids_and_clusters(self):
        db = self.get_db()[:]

        # Initialize the medoids randomly
        self.init_medoids()

        # Create by value a local list of medoids
        medoids = self.get_medoids()[:]

        # TODO: Remove if not needed
        # Holds the number of examples contained in each cluster
        # size_of_clust = [0 for i in range(self.get_k())]

        # Holds which cluster each example belongs to
        ex_to_clust = [-1 for i in range(len(self.get_db()))]

        # Each cluster is a list of examples, including the medoid
        clusters = [[] for i in range(self.get_k())]

        # We will iterate an arbitrary number of times at most
        for curr_iter in range(self.get_max_iters()):
            
            # Find the closest medoid for each example
            for curr_ex_idx in range(len(self.get_db())):
                
                # Get the medoid closest to our current example
                med_idx = self.assign_ex(self.get_db()[curr_ex_idx], False)

                # Assign that example to its closest medoid
                ex_to_clust[curr_ex_idx] = med_idx

                # Add the example (and its db index) to the cluster
                # represented by its respective medoid
                clusters[med_idx].append( \
                    [deepcopy(self.get_db()[curr_ex_idx]), \
                    curr_ex_idx])
                
            # Now we need to find some new medoids in each cluster
            # (see for loop directly below)
            new_medoids = [-1 for i in range(len(medoids))]
            
            # For each medoid...
            for med_idx in range(len(medoids) - 1):
                
                # Find the example that is the "best medoid" for each cluster.
                # Best meaning it minimizes the sum of
                # distances from all other points within the cluster
                new_medoids[med_idx] = self.update_medoid(clusters[med_idx], med_idx)
            
            # See if the medoids changed
            medoids_were_changed = False
            for med_idx in range(len(new_medoids) - 1):

                # If the medoids changed... 
                if new_medoids[med_idx] != medoids[med_idx]:

                    # Update the list of medoids
                    medoids[med_idx] = new_medoids[med_idx]

                    # Change the boolean to True
                    medoids_were_changed = True
            
            # If the medoids did change...
            if medoids_were_changed is True:

                # Reset the clusters so we can re-assign examples
                # to new medoids in the next iteration
                clusters = [[] for i in range(self.get_k())]
            
            # Otherwise, we have our final medoids
            # and our clusters and we can stop
            else:
                self.set_medoids(medoids)
                
                clusters_without_idxs = [[] for i in range(len(medoids))]

                for cluster_idx, cluster in enumerate(clusters):
                    for ex in cluster:
                        clusters_without_idxs[cluster_idx].append(ex[0])

                self.set_kmedoids_clusters(clusters_without_idxs)
                return
        
        # We have ran out of iterations,
        # so we set the medoids and clusters
        self.set_medoids(medoids)
                
        clusters_without_idxs = []

        for cluster in clusters:
            for ex in cluster:
                clusters_without_idxs.append(ex[0])

        self.set_kmedoids_clusters(clusters_without_idxs)

    """ ---------------------------------------------------
    TODO: add info
    """
    def calc_kmeans_clusters(self):
        
        # Initialize list of clusters
        clusters = [[] for i in range(self.get_k())]

        for ex in self.get_db()[:]:
            cent_idx = self.assign_ex(ex, True)
            clusters[cent_idx].append(ex)
        
        for c in clusters:
            if len(c) == 0:
                clusters.remove(c)

        self.set_kmeans_clusters(clusters)

    """ ---------------------------------------------------
    TODO: add info
    """
    def update_medoid(self, cluster, med_idx):
        
        # The minimum within-cluster sum of squares
        min_wcss = sys.maxsize

        # The db index of the example that yields the
        # minimum within-cluster sum of squares
        min_wcss_idx = -1

        # A temp variable to hold an iteration's
        # within-cluster sum of squares
        curr_wcss = 0

        # For every example in the cluster...
        for curr_idx, curr in enumerate(cluster):

            # Find the wcss from the current example
            # to every other example
            for other_idx, other in enumerate(cluster):
                if (other_idx != curr_idx):
                    curr_wcss += self.euc_dist(curr[0], other[0])

            # Set the new wcss if the new wcss is lower
            if curr_wcss < min_wcss:
                min_wcss = deepcopy(curr_wcss)
                min_wcss_idx = curr[1]
            
            # Reset the temp within-cluster sum of squares
            curr_wcss = 0
        
        # Return the db index of our new medoid
        return min_wcss_idx

    # ---------------------------------------------------
    # Setters

    def set_centroids(self, in_centroids):
        self.centroids = in_centroids
    
    def set_medoids(self, in_medoids):
        self.medoids = in_medoids

    def set_kmeans_clusters(self, in_clusters):
        self.kmeans_clusters = in_clusters

    def set_kmedoids_clusters(self, in_clusters):
        self.kmedoids_clusters = in_clusters

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
    
    def get_medoids(self):
        return self.medoids
    
    def get_kmeans_clusters(self):
        return self.kmeans_clusters

    def get_kmedoids_clusters(self):
        return self.kmedoids_clusters

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
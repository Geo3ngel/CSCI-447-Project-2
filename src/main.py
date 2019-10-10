""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""

import os
import process_data
from knn import knn
from kcluster import kcluster as kc
from path_manager import pathManager as path_manager
import validate
import statistics

debug_file = open("../output_files/debug_output.txt", "w")
output_file = open("../output_files/output_file.txt", "w")

# Asks for user to select a database from a list presented from current database collection directory.
def select_db(databases):  

    if len(databases) == 0:
        print("ERROR: No databases found!")
        return False
    
    chosen = False
    db = ""
    
    # Selection loop for database
    while(not chosen):
        print("\nEnter one of the databases displayed:", databases)
        db = input("Entry: ")
        print("database:", db)
        if db in databases:
            print("Selected:", db)
            chosen = True
        else:
            print(db, "is an invalid entry. Try again.")
    return db

# Cleaner print outs for the sake of my sanity.
def print_db(db):
    
    if len(db) < 1:
        print("[] - Empty")
    else:
        for row in db:
            print(row)

# Initializes path manager with default directory as databases.
pm = path_manager()

# Loads in a list of database folders
# for the user to select as the current database.
# selected_db = select_db(pm.find_folders(pm.get_databases_dir()))

# Set of each dataset
datasets = [
    ['abalone', 'classification']
]

output_file.write('CURRENT DATASET: ' + datasets[0][0] + '\n')
debug_file.write('CURRENT DATASET: ' + datasets[0][0] + '\n')

output_file.write('DATA TYPE: ' + datasets[0][1] + '\n')
debug_file.write('DATA TYPE: ' + datasets[0][1] + '\n')

# Sets the selected database folder
# in the path manager for referencing via full path.
pm.set_current_selected_folder(datasets[0][0])

# Processes the file path of the database into
# a pre processed database ready to be used as a learning/training set.
db = process_data.process_database_file(pm)

# Sanity checks.
normal_data, irregular_data = process_data.identify_missing_data(db)
corrected_data = process_data.extrapolate_data(normal_data, irregular_data, db.get_missing_symbol())

# repaired_db is the total database once the missing values have been filled in.
if len(corrected_data) > 0:
    repaired_db = normal_data + corrected_data
else:
    repaired_db = normal_data
    
db.set_data(repaired_db)

# Convert the discrete data to type float.
db.convert_discrete_to_float()

debug_file.write('\n\nFULL DATASET: \n')
for row in db.get_data():
    debug_file.write(str(row) + '\n')

# COMMENTING THIS OUT AS WE DON'T WANT DISCRETIZED DATA AT THIS POINT IN TIME
# process_data.convert(db.get_data())

# -------------------------------------------------------------
# k-nearest neighbors

# print('\nRUNNING K-NEAREST NEIGHBORS\n')
# knn_predicted = knn.k_nearest_neighbors(5, \
#                                     'classification', \
#                                     db.get_training_data(0, 99), \
#                                     db.get_data()[107], \
#                                     db.get_classifier_col(), \
#                                     db.get_classifier_attr_cols())

# print(knn_predicted)

# -------------------------------------------------------------
# editied k-nearest neighbors

# print('\nRUNNING EDITED K-NEAREST NEIGHBORS\n')
# eknn_predicted = knn.edited_knn(5, \
#                'classification', \
#                db.get_training_data(0, 100), \
#                db.get_classifier_col(), \
#                db.get_classifier_attr_cols())

# print(eknn_predicted)

# -------------------------------------------------------------
# Condensed nearest neighbors

# print('\nRUNNING CONDENSED NEAREST NEIGHBORS\n')
# cnn_predicted = knn.condensed_nn(db.get_training_data(0,100), \
#                                  db.get_classifier_col(), \
#                                  db.get_classifier_attr_cols())

# print(cnn_predicted)

# -------------------------------------------------------------
# k-means clustering and k-medoids clustering

# print('\nRUNNING K-MEANS CLUSTERING')
# kc = kcluster(5, 300, 0.01)

# print('\nk_means.get_centroids()')
# print(kc.get_centroids())

# for idx, cluster in enumerate(kc.get_kmeans_clusters()):
#     print('\nk_means.get_clusters()[' + str(idx) + ']')
#     print(cluster)

# print('\nk_means.get_medoids()')
# print(kc.get_medoids())

# for idx, cluster in enumerate(kc.get_kmedoids_clusters()):
#     print('\nk_medoids.get_clusters()[' + str(idx) + ']')
#     print(cluster)

# -------------------------------------------------------------
# k-fold cross validation

# trainSet = [[2,2,2],[4,4,4]]
# testInstance = [5,5,5]
# k = 1

# print_db(db.get_data())
print("RUNNING K-FOLD CROSS VALIDATION")

# Prepare data for k-fold
binned_data, bin_lengths = process_data.separate_data(db.get_attr(), db.get_data())
# Extract validation set
bin_lengths, validate_data, binned_data = validate.get_validate(bin_lengths, binned_data)

debug_file.write('\n\nVALIDATION DATA: \n')
for row in validate_data:
    debug_file.write(str(row) + '\n')


knn = knn(5, datasets[0][1], db.get_classifier_col(), db.get_classifier_attr_cols())

#NOTE binned_data needs to still be shuffled somewhere above here
#NOTE this bitch will probs have to use 9 instead of 10 because we are removing a bin from bin_lengths
validate.k_fold(9, binned_data, \
                validate_data, bin_lengths, db, \
                False, datasets[0][1], \
                knn, debug_file, output_file, 'edited_nn')


# print(knn.k_nearest_neighbors(trainSet, testInstance))

# print(knn.get_k_nearest_neighbors(trainSet, testInstance, 1))
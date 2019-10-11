""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""

import os
import process_data
from knn import knn
from kcluster import kcluster
from path_manager import pathManager as path_manager
import validate
import statistics
from copy import deepcopy

debug_file = open("debug_output.txt", "w")
output_file = open("output_file.txt", "w+")

# Asks for user to select a database from a list presented from current database collection directory.
def select_db(databases):  

    if len(databases) == 0:
        print("ERROR: No databases found!")
        return False
    
    chosen = False
    db = ""
    chosen_dbs = []
    
    # Selection loop for database
    while(not chosen):
        print("\nEnter one of the databases displayed, or 'all' to run for all avalible databases.:", databases)
        db = input("Entry: ")
        print("database:", db)
        if db in databases:
            print("Selected:", db)
            chosen_dbs.append(db)
            chosen = True
        elif db.lower() == "all":
            print("Running for all Databases.")
            chosen_dbs = ["abalone", "car", "forestfires", "machine", "segmentation", "wine"]
            chosen = True
        else:
            print(db, "is an invalid entry. Try again.")
    return chosen_dbs

# Cleaner print outs for the sake of my sanity.
def print_db(db):
    
    if len(db) < 1:
        print("[] - Empty")
    else:
        for row in db:
            print(row)

def prepare_db(database, pm):
    # Sets the selected database folder
    # in the path manager for referencing via full path.
    pm.set_current_selected_folder(database)
    # Processes the file path of the database into
    # a pre processed database ready to be used as a learning/training set.
    db = process_data.process_database_file(pm)

    output_file.write('CURRENT DATASET: ' + database + '\n')
    debug_file.write('CURRENT DATASET: ' + database + '\n')
    output_file.write('DATA TYPE: ' + db.get_dataset_type() + '\n')
    debug_file.write('DATA TYPE: ' + db.get_dataset_type() + '\n')
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
    # TODO: make it append the database name to the debug file aswell, so we can get every dataset when running for all of them.
    debug_file.write('\n\nFULL DATASET: \n')
    for row in db.get_data():
        debug_file.write(str(row) + '\n')
    
    return db

def main_execution():
    # Sets to store all the loss func avs
    k_nn_classification_avgs = []
    k_nn_regress_avgs = []
    enn_avgs = []
    cnn_avgs = []
    k_means_classification_avgs = []
    k_means_regress_avgs = []
    k_medoid_classification_avgs = []
    k_medoid_regress_avgs = []
    
    reduction_funcs = [
        # 'edited_nn',
        # 'condensed_nn',
        'k_means',
        'k_medoids'
    ]

    # Initializes path manager with default directory as databases.
    pm = path_manager()

    # Loads in a list of database folders
    # for the user to select as the current database.
    selected_dbs = select_db(pm.find_folders(pm.get_databases_dir()))
    # TODO: change to get dataset type from db
    for database in selected_dbs:
        db = prepare_db(database, pm)
        k_nearest = knn(5, db.get_dataset_type(), \
                db.get_classifier_col(), \
                db.get_classifier_attr_cols())
        # Start k-fold cross validation
        print("RUNNING K-FOLD CROSS VALIDATION")
        # Prepare data for k-fold
        binned_data, bin_lengths = process_data.separate_data(db.get_attr(), db.get_data())
        # Extract validation set
        bin_lengths, validate_data, binned_data = validate.get_validate(bin_lengths, binned_data)
        debug_file.write('\n\nVALIDATION DATA: \n')
        for row in validate_data:
            debug_file.write(str(row) + '\n')
            #NOTE binned_data needs to still be shuffled somewhere above here

        # Run k-fold on just k-means first
        k_fold_results = validate.k_fold(9, binned_data, \
                                        validate_data, bin_lengths, \
                                        db, False, db.get_dataset_type(), \
                                        k_nearest, debug_file, output_file,)
        
        if db.get_dataset_type() == 'classification':
            k_nn_classification_avgs.append(sum(k_fold_results) / len(k_fold_results))
        elif db.get_dataset_type() == 'regression':
            k_nn_regress_avgs.append(sum(k_fold_results) / len(k_fold_results))

        output_file.write('\n\n\n')
        
        # Tuning
        if True:
            
            # Attributes to be removed
            removal_queue = []
            removed_attr_idx = None
            
            norm_sum = sum(k_fold_results) / len(k_fold_results)
            for attr_idx in db.get_classifier_attr_cols():
                
                # Stores full classifier attributes list.
                temp_db = deepcopy(k_nearest.get_class_cols())[:]
                
                # Recomputes k-fold cross validation
                # Sets databae classifier attributes idx list to shorter version temporarily.
                tmp = k_nearest.get_class_cols()
                tmp.remove(attr_idx)
                k_nearest.set_class_cols(tmp)
                
                # recomputes the k fold results for comparison
                # Prepare data for k-fold
                binned_data, bin_lengths = process_data.separate_data(db.get_attr(), db.get_data())
                # Extract validation set
                bin_lengths, validate_data, binned_data = validate.get_validate(bin_lengths, binned_data)
             
                # Run k-fold on just k-means first
                k_fold_results = validate.k_fold(9, binned_data, \
                                                validate_data, bin_lengths, \
                                                db, False, db.get_dataset_type(), \
                                                k_nearest, debug_file, output_file,)
                
                if db.get_dataset_type() == 'classification':
                    k_nn_classification_avgs.append(sum(k_fold_results) / len(k_fold_results))
                elif db.get_dataset_type() == 'regression':
                    k_nn_regress_avgs.append(sum(k_fold_results) / len(k_fold_results))

                attr_removed_sum  = sum(k_fold_results) / len(k_fold_results)
                
                # Resets the database column set.
                print(k_nearest.get_class_cols())
                k_nearest.set_class_cols(temp_db)
                
                # Remove attr_idx from the removal queue if the accuracy is worse.
                print("COMPARISON FOR:", attr_idx, ", VALUES:", norm_sum, ">", attr_removed_sum)
                print(k_nearest.get_class_cols())
                if norm_sum < attr_removed_sum:
                    # Add to removal queue.
                    removal_queue.append(attr_idx)
            
            final_attrs = k_nearest.get_class_cols()
            
            print("ATTRINBUTE IDX:", removal_queue)
            
            # Removes the attribute idx values from the database's list of attribute columns to be used.
            for attr_idx in removal_queue:
                final_attrs.remove(attr_idx)
                
            db.set_classifier_attr_cols(final_attrs)
                
            print(database, " is using attribute colums: ", db.get_classifier_attr_cols())

main_execution()
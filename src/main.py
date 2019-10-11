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

debug_file = open("machine_debug_output.txt", "w")
output_file = open("machine_output_file.txt", "w+")

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
        'edited_nn',
        'condensed_nn',
        'k_means',
        'k_medoids'
    ]

    # Initializes path manager with default directory as databases.
    pm = path_manager()

    # Loads in a list of database folders
    # for the user to select as the current database.
    selected_dbs = select_db(pm.find_folders(pm.get_databases_dir()))
    for database in selected_dbs:
        db = prepare_db(database, pm)
        k_nearest = knn(15, db.get_dataset_type(), \
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

        # Run k-fold on just k-means first
        k_fold_results = validate.k_fold(9, binned_data, \
                                        validate_data, bin_lengths, \
                                        db, True, db.get_dataset_type(), \
                                        k_nearest, debug_file, output_file,)
        
        if db.get_dataset_type() == 'classification':
            k_nn_classification_avgs.append(sum(k_fold_results) / len(k_fold_results))
        elif db.get_dataset_type() == 'regression':
            k_nn_regress_avgs.append(sum(k_fold_results) / len(k_fold_results))

        output_file.write('\n\n\n')
        
        # Loop thru all reduction functions
        for func in reduction_funcs:
            print('RUNNING ', func)
            # we are removing a bin from bin_lengths
            
            if db.get_dataset_type() == 'classification':
                k_fold_results = validate.k_fold(9, binned_data, \
                                                validate_data, bin_lengths, db, \
                                                True, db.get_dataset_type(), \
                                                k_nearest, debug_file, output_file, func)
            
                if func == 'edited_nn':
                    enn_avgs.append(sum(k_fold_results) / len(k_fold_results))
                elif func == 'condensed_nn':
                    cnn_avgs.append(sum(k_fold_results) / len(k_fold_results))
                elif func == 'k_means':
                    k_means_classification_avgs.append(sum(k_fold_results) / len(k_fold_results))
                elif func == 'k_medoids':
                    k_medoid_classification_avgs.append(sum(k_fold_results) / len(k_fold_results))
            
            elif db.get_dataset_type() == 'regression':
                if func == 'edited_nn' or func == 'condensed_nn':
                    continue
                
                # Shrink data to quarter of the size
                db_small = process_data.random_data_from(db.get_data(), 0.25)
                # Re-bin it after shrinking
                binned_data, bin_lengths = process_data.separate_data(db.get_attr(), db_small) 

                k_fold_results = validate.k_fold(9, binned_data, \
                                                validate_data, bin_lengths, db, \
                                                True, db.get_dataset_type(), \
                                                k_nearest, debug_file, output_file, func)
                
                if func == 'k_means':
                    pass
                elif func == 'k_medoids':
                    pass
                
            
            output_file.write('K FOLD RESULTS: ' + str(k_fold_results))
            
            output_file.write('\n\n\n')
            
        print("KNN CLASSIFICATION AVGS: ", k_nn_classification_avgs)
        print("KNN REGRESSION AVGS: ", k_nn_regress_avgs)
        print("ENN AVGS: ", enn_avgs)
        print("CNN AVGS: ", cnn_avgs)
        print("K MEANS CLASSIFICATION AVGS: ", k_means_classification_avgs)
        print("K MEANS REGRESSION AVGS: ", k_means_regress_avgs)
        print("K MEDOID CLASSIFICATION AVGS: ", k_medoid_classification_avgs)
        print("K MEDOID REGRESSION AVGS", k_medoid_regress_avgs)
    




main_execution()
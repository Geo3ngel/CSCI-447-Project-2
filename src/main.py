""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""

import os
import process_data
import knn
from path_manager import pathManager as pm
import statistics

# Asks for user to select a database from a list presented from current database collection directory.
def select_database(databases):
    
    if len(databases) == 0:
        print("ERROR: No databases found!")
        return False
    
    chosen = False
    database = ""
    
    # Selection loop for database
    while(not chosen):
        print("\nEnter one of the databases displayed:", databases)
        database = input("Entry: ")
        if database in databases:
            print("Selected:", database)
            chosen = True
        else:
            print(database, "is an invalid entry. Try again.")
        
    return database

# Cleaner print outs for the sake of my sanity.
def print_database(database):
    
    if len(database) < 1:
        print("[] - Empty")
    else:
        for row in database:
            print(row)

# Initializes path manager with default directory as databases.
path_manager = pm()

# Loads in a list of database folders for the user to select as the current database.
selected_database = select_database(path_manager.find_folders(path_manager.get_databases_dir()))

# Sets the selected database folder in the path manager for referencing via full path.
path_manager.set_current_selected_folder(selected_database)

# Processes the file path of the database into a pre processed database ready to be used as a learning/training set.
db = process_data.process_database_file(path_manager)

# Sanity checks.
normal_data, irregular_data = process_data.identify_missing_data(db)

corrected_data = process_data.extrapolate_data(normal_data, irregular_data, db.get_missing_symbol())

# repaired_db is the total database once the missing values have been filled in.
if len(corrected_data) > 0:
    repaired_db = normal_data + corrected_data
else:
    repaired_db = normal_data
    
db.set_data(repaired_db)

# process_data.convert(db.get_data())

knn.get_nearest_neighbors(5, db.get_training_data(0, 10), db.get_data()[11], db.get_classifier_col(), db.get_classifier_attr_cols())
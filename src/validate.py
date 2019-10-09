""" -------------------------------------------------------------
@file        k_fold_cross_validation.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       Contains all functionality related to k-fold cross-validation
"""

from copy import deepcopy
# from loss_functions import *
import process_data

""" -------------------------------------------------------------
@param  k                   The number of folds we are using for k-fold cross validation
@param  binned_data_set     A list of examples/samples of a database repository
@param  bin_lengths         A list containing the lengths of each bin, the index in the list is
@param  type                The type of dataset (either classification or regression)
@param reduction_func       The function we will use to reduce our training set

@return    binned_guess_results: [[<incorrect_guesses>, <correct_guesses>]]
                incorrect_guesses: [[expected answer,incorrect guess]]
                correct_guesses: [correct guess] 

@brief     Given a number of folds k, and binned data, as well as bin_lengths,
           iterate over each bin and separate the data into two subsets, 
           training data and test_data. 
           Then classify the training data and calculate the probabilities.
           Run prediction on each row of the test_data set and return
           an array of guess results containing
           guess results associated with each bin,
           as every bin will be the test bin at some point.
"""

import knn
def k_fold(k, binned_data_set, bin_lengths, db, shuffle, type, knn, reduction_func = None):
    # List to store mean abs error from all k iterations of any regression dataset
    mae_results = []
    # List to store 0-1 loss results from all k iterations or any classification dataset
    loss_results = [] 
    attr_headers = db.get_attr()
    class_list = db.get_classifiers()
    # For each bin in our data
    for bin_number in range(k):
        print("BIN NUMBER: ", bin_number)
        test_data = []
        training_data = deepcopy(binned_data_set)
        row_idx = 0
        
        # Add rows from the main data set to our test_data subset 
        # until it is the length of the bin that we are using as our 
        # test bin (This is to ensure we stop after finding all of the 
        # rows that match the bin we want to use)
        while len(test_data) < bin_lengths[bin_number]:
            if training_data[row_idx][0] == bin_number:
                test_data.append(training_data.pop(row_idx).copy()[1:])
                row_idx -=1
            row_idx += 1

        # Remove the bin numbers from our training data, this is done because our classifier does not support bin numbers
        for row_idx2 in range(len(training_data)):
            training_data[row_idx2].pop(0)
            training_data[row_idx2] = training_data[row_idx2][0]

        if shuffle:
            training_data = process_data.shuffle_all(training_data,.1)
        
        # Classify our training data set, so we can calculate the probabilites
        # classified_training_data = classifier.classify_db(attr_headers, training_data, db.get_classifier_col())

        # Calculate the probabilities
        # training_probs = classifier.calc_prob_of_response(classified_training_data)

        loss_results = [] # Set of each 0-1 loss result
        abs_errors = [] # Set of absolute errors of each regression prediction
        # For each row (sample) in our test_data, run knn to predict its class

        for test_row in test_data:
            # TODO: implement way to pass in the correct 
            # classifying function (edited_nn, condensed_nn, etc) as parameter
            if reduction_func:
                training_data = reduction_func(training_data)
            # Guess class with knn
            predicted = knn.k_nearest_neighbors(training_data, test_row)
            
            if type == 'classification':
                if predicted == test_row[db.get_classifier_col()]:
                    loss_results.append(0)
                else:
                    loss_results.append(1)

            elif type == 'regression':
                abs_errors.append(abs(float(test_row[db.get_classifier_col()]) - predicted))
        
        # Compute average 0-1 loss and mean absolute error for this iteration
        if type == 'classification':
            loss_results.append(sum(loss_results) / len(loss_results))
        elif type == 'regression':
            mae_results.append(sum(abs_errors) / len(abs_errors))
    
    print("0-1 LOSS RESULTS: ", loss_results)
    print("MAE RESULTS: ", mae_results)

        
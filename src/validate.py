""" -------------------------------------------------------------
@file        k_fold_cross_validation.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       Contains all functionality related to k-fold cross-validation
"""

from copy import deepcopy
import process_data
import random
from kcluster import kcluster

'''
@brief  Create a validation data set to be used with
        edited_nn. For now we will just select random rows.
        The size of the validation set will be the size of 
        one bin.
'''
def get_validation_data(db, bin_length):
    validation_data = []
    used_idxs = []
    while len(validation_data) < bin_length:
        idx = random.randint(0,len(db.get_data())-1)
        if idx in used_idxs:
            continue
        else:
            validation_data.append(db.get_data()[idx])
            used_idxs.append(idx)
    return validation_data

    

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

def get_validate(bin_lengths, binned_data_set):
    new_bin_lengths = bin_lengths
    validate_set = []
    new_data_set = binned_data_set
    for idx in range(bin_lengths[len(bin_lengths)-1]):
        validate_set.append(new_data_set.pop()[1])
    new_bin_lengths.pop()
    return new_bin_lengths, validate_set, new_data_set

def k_fold(k, binned_data_set, validate_data, bin_lengths, db, shuffle, type, knn, debug_file, output_file, reduction_func = None):
    # List to store mean abs error from all k iterations of any regression dataset
    debug_file.write('STARTING K-FOLD\n')
    output_file.write('STARTING K-FOLD\n')
    
    if reduction_func:
        debug_file.write('RUNNING WITH ' + reduction_func + '\n')
        output_file.write('RUNNING WITH ' + reduction_func + '\n')

    mse_results = []
    # List to store 0-1 loss results from all k iterations or any classification dataset
    loss_results = [] 
    attr_headers = db.get_attr()
    class_list = db.get_classifiers()
    # For each bin in our data
    for bin_number in range(k):
        print("K FOLD ITERATION: ", bin_number)
        output_file.write('K FOLD ITERATION ' + str(bin_number) + '\n')
        debug_file.write('K FOLD ITERATION ' + str(bin_number) + '\n')

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
        
        debug_file.write('TRAINING DATA: \n')
        for row in training_data:
            debug_file.write(str(row) + '\n')
        print('FULL TRAINING DATA LENGTH: ', len(training_data))
        # Check which reduction_func we are using
        if reduction_func == 'edited_nn':
            training_data = knn.edited_knn(training_data, validate_data)
            debug_file.write('\n\n REDUCED TRAINING DATA: \n')
            for row in training_data:
                debug_file.write(str(row) + '\n')
        elif reduction_func == 'condensed_nn':
            training_data = knn.condensed_nn(training_data)
            debug_file.write('\n\n REDUCED TRAINING DATA: \n')
            for row in training_data:
                debug_file.write(str(row) + '\n')
        elif reduction_func == 'k_means':
            if type == 'classification':
                edited_data = knn.edited_knn(training_data, validate_data)
                print("Finished enn.")
                print("Making ", len(edited_data), " clusters.")
                kc = kcluster(len(edited_data), 100, training_data, db.get_classifier_attr_cols(), 'k-means')
            else:
                num_clusters = math.sqrt(len(training_data))
                kc = kcluster(num_clusters, 100, training_data, db.get_classifier_attr_cols(), 'k-means')
            training_data = kc.get_centroids()

        elif reduction_func == 'k_medoids':
            if type == 'classification':
                edited_data = knn.edited_knn(training_data, validate_data)
                print("Finished enn.")
                print("Making ", len(edited_data), " clusters.")
                kc = kcluster(len(edited_data), 100, training_data, db.get_classifier_attr_cols(), 'k-medoids')
            else:
                num_clusters = math.sqrt(len(training_data))
                kc = kcluster(num_clusters, 100, training_data, db.get_classifier_attr_cols(), 'k-medoids')
            medoid_idxs = kc.get_medoids()
            new_training_data = []
            for idx in medoid_idxs:
                new_training_data.append(training_data[idx])
            training_data = new_training_data


        
        print('CONDENSED TRAINING DATA LENGTH: ', len(training_data))
        
        current_loss_results = [] # Set of each 0-1 loss result
        squared_errors = [] # Set of absolute errors of each regression prediction
        
        debug_file.write('\n\nTEST DATA: \n')
        for row in test_data:
            debug_file.write(str(row) + '\n')
        # For each row (sample) in our test_data, run knn to predict its class
        for test_row in test_data:
            debug_file.write('RUNNING K-NN ON TEST POINT ' + str(test_row) + '\n')
            # Guess class with knn
            predicted = knn.k_nearest_neighbors(training_data, test_row[0])
            debug_file.write('PREDICITED CLASS: ' + str(predicted) + '\n')
            if type == 'classification':
                if predicted == test_row[0][db.get_classifier_col()]:
                    current_loss_results.append(0)
                else:
                    current_loss_results.append(1)

            elif type == 'regression':
                squared_errors.append(pow((float(test_row[0][db.get_classifier_col()]) - predicted), 2))
        
        # Compute average 0-1 loss and mean absolute error for this iteration
        if type == 'classification':
            loss = sum(current_loss_results) / len(current_loss_results)
            output_file.write('CALCULATED LOSS: ' + str(loss) + '\n')
            debug_file.write('CALCULATED LOSS: ' + str(loss) + '\n')
            loss_results.append(loss)
        elif type == 'regression':
            mse = sum(squared_errors) / len(squared_errors)
            output_file.write('CALCULATED MsE: ' + str(mse) + '\n')
            output_file.write('CALCULATED MsE: ' + str(mse) + '\n')
            mse_results.append(mse)
    
    print("0-1 LOSS RESULTS: ", loss_results)
    print("MsE RESULTS: ", mse_results)
    # Return the correct loss function results
    if type == 'classification':
        return loss_results
    else:
        return mse_results
        
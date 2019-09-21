""" -------------------------------------------------------------
@file       database.py
@authors    George Engel, Troy Oster, Dana Parker, Henry Soule
@brief      Object that stores the information of
            each data repository for ease of access & manipulation
"""

class database:
    """
    @param  data_array  List of data from one data repository
                        that will be or has been filtered.
    """
    def __init__(self, data_array):
        print("Database initialized.")
        self.data = data_array
        
    def to_string(self):
        print(self.data)
        
    def get_data(self):
        return self.data
    
    def set_data(self, data_array):
        self.data = data_array
        
    def get_attr(self):
        return self.attributes
    
    def get_classifier_col(self):
        return self.classifier_column
    
    def get_classifier_attr_cols(self):
        return self.classifier_attr_columns
    
    def get_missing_symbol(self):
        return self.missing_symbol
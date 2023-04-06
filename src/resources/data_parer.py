# 
# Data Parsing Function
# By: Cameron Kaminski
# 
#
# This function takes the file path to a .txt file, returns the data as a numpy
# matrix.
#

import os
import numpy as np

def readDataToMatrix(file_name, targets=True, type = 'C'):
    """
    Takes the file path to a .txt file, returns the data as a numpy matrix.
    @param file_path : str
    @param targets : bool
    @param type : chr
    @return numpy.ndarray
    """

    grand_parent_dir = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), os.pardir))

    path = os.path.join(grand_parent_dir, 'prog1_data', file_name)

    with open(path, 'r') as file:
        lines = file.readlines()
    if targets:
        if type == 'C':
            data_matrix = np.array([[int(val) for val in line.strip().split()]
                for line in lines])
        elif type == 'R':
            data_matrix = np.array([[float(val) for val in line.strip().split()]
                for line in lines])
    else:
        data_matrix = np.array([[float(val) for val in line.strip().split()]
            for line in lines])                                               
    
    return data_matrix
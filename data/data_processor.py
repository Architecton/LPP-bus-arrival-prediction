import numpy as np
import csv

"""
Ideas for feature creation:
date -> hours, minutes, seconds
date -> is_weekend, is_holiday
route direction -> 0/1 (depending on direction)
"""


class DataProcessor:

    # Constructor: initialize instance with file containing the data.
    def __init__(self, data_file):
        # Create matrix out of
        reader = csv.reader(open(data_file, "r"), delimiter="\t")
        x = list(reader)
        self.data_matrix = np.array(x)
        self.bus_line_matrices = None


    # _get_bus_line_matrices: make a dictionary that maps each bus line to a matrix of its
    # features obtained from the csv file.
    def _get_bus_line_matrices(self):

        # add_line_row: add line in data matrix as value for corresponding key in dictionary.
        def add_line_row(x):
            self.bus_line_matrices[x[2]] = np.vstack((self.bus_line_matrices[x[2]], x))

        # Get unique bus lines and make a dictionary that maps bus lines to empty row vector.
        self.bus_line_matrices = dict.fromkeys(np.unique(self.data_matrix[1:, 2]), np.array([], dtype=np.character).reshape(0, self.data_matrix.shape[1]))

        np.fromiter((add_line_row(row) for row in self.data_matrix[1:,:]), self.data_matrix.dtype, count=len(self.data_matrix[1:,:]))

    # _decompose_features: construct new features from the composite features found in the data file.
    def _decompose_features(self):
        # TODO create a new class feature engineering with methods that append features to matrices
        pass


if __name__ == '__main__':
    # dp = DataProcessor('test.csv')
    # dp._get_bus_line_matrices()
    pass
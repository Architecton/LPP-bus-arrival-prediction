import numpy as np
import csv
from lib_naloga3 import feature_eng
import os
import pdb

# DataProcessor: Class implementing methods used to create a feature matrix that can be used for regression analysis.
class DataProcessor:

    # Constructor: initialize instance with file containing the data.
    def __init__(self, data_file):
        # Create numpy matrix out of the LPP data sheet
        reader = csv.reader(open(data_file, "r"), delimiter="\t")
        x = list(reader)
        self.data_matrix = np.array(x)

        # bus_line_matrices maps bus line numbers to their matrices obtained from the LPP data sheet.
        self.bus_line_matrices = dict()

        # bus_line_feature_matrices maps bus lines to their processed matrices that can be used for linear regression.
        self.bus_line_feature_matrices = dict()


    # _get_bus_line_matrices: make a dictionary that maps each bus line to a matrix of its
    # features obtained from the csv file.
    def _get_bus_line_matrices(self):

        # add_line_row: add line in data matrix as value for corresponding key in dictionary.
        def add_line_row(x):
            self.bus_line_matrices[x[2]] = np.vstack((self.bus_line_matrices[x[2]], x))

        # Get unique bus lines and make a dictionary that maps bus lines to empty row vector.
        self.bus_line_matrices = dict.fromkeys(np.unique(self.data_matrix[1:, 2]), np.array([], dtype=np.character).reshape(0, self.data_matrix.shape[1]))

        np.fromiter((add_line_row(row) for row in self.data_matrix[1:,:]), self.data_matrix.dtype, count=len(self.data_matrix[1:,:]))


    # _engineer_features: construct new features from the composite features found in the data file and add to data matrix.
    # Also add weather features obtained from ARSO website.
    def _engineer_features(self):
        # Go over bus lines and construct feature matrices that are appropriate for regression analysis.
        for bus_line in self.bus_line_matrices.keys():
            self.bus_line_feature_matrices[bus_line] = feature_eng.get_feature_matrix_training(self.bus_line_matrices[bus_line])


# Run data processing to get data.
if __name__ == '__main__':
    # Create DataProcessor instances and initialize with training and test data.
    dp1 = DataProcessor('competition_data/train.csv')
    dp1._get_bus_line_matrices()

    dp2 = DataProcessor('competition_data/test.csv')
    dp2._get_bus_line_matrices()

    # Create files containing data for training and test data.
    for line_name in dp1.bus_line_matrices.keys():
        nxt = dp1.bus_line_matrices[line_name]
        features, target, start_times = feature_eng.get_feature_matrix_train(nxt)
        os.mkdir('bus-line-data-train/' + line_name)
        np.save('bus-line-data-train/' + line_name + '/predictors_train.npy', features)
        np.save('bus-line-data-train/' + line_name + '/target_train.npy', target)
        np.save('bus-line-data-train/' + line_name + '/start_times_train.npy', start_times)

    for line_name in dp2.bus_line_matrices.keys():
        nxt = dp2.bus_line_matrices[line_name]
        features, start_times = feature_eng.get_feature_matrix_test(nxt)
        os.mkdir('bus-line-data-test/' + line_name)
        np.save('bus-line-data-test/' + line_name + '/predictors_test.npy', features)
        np.save('bus-line-data-test/' + line_name + '/start_times_test.npy', start_times)
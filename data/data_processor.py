"""
Ideas for feature creation:
date -> hours, minutes, seconds
date -> is_weekend, is_holiday
route direction -> 0/1 (depending on direction)
"""

class data_processor:

    # Constructor: initialize instance with file containing the data.
    def __init__(self, data_file):
        self.data = data_file
        self.bus_line_matrices = None

    # _extract_line_numbers: extract all bus line numbers found in csv data file.
    def _extract_line_numbers(self):
        # TODO
        pass

    # _get_bus_line_matrices: make a dictionary that maps each bus line to a matrix of its
    # features obtained from the csv file.
    def _get_bus_line_matrices(self):
        # TODO
        pass

    # _decompose_features: construct new features from the composite features found in the data file.
    def _decompose_features(self):
        # TODO create a new class feature engineering with methods that append features to matrices
        pass
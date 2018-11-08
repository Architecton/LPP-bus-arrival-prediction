import datetime
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# decompose_datetime: take the date column and create a matrix containing the day, hour, minutes and seconds as columns.
# Also create a column vector of datetime objects for the data.
def decompose_datetime(datetime_col):
    # Define string for parsing datetime.
    datetime_parse_str = "%Y-%m-%d %H:%M:%S.%f"

    # Allocate matrix for storing the results.
    res = np.empty([len(datetime_col), 13], dtype=int)
    dt_col = np.empty((len(datetime_col), 1), dtype=object)
    # Initialize row counter.
    r_count = 0
    for dt in datetime_col:
        date_obj = datetime.datetime.strptime(dt, datetime_parse_str)
        # month
        res[r_count, 0] = date_obj.date().month
        # day
        res[r_count, 1] = date_obj.date().day
        # hour
        res[r_count, 2] = date_obj.time().hour
        # minute
        res[r_count, 3] = date_obj.time().minute
        # second
        res[r_count, 4] = date_obj.time().second
        # is saturday (days are numbered from 0 to 6)
        res[r_count, 5] = 1 if date_obj.weekday() == 5 else 0
        # is sunday (days are numbered from 0 to 6)
        res[r_count, 6] = 1 if date_obj.weekday() == 6 else 0
        # day of week
        res[r_count, 7] = date_obj.weekday()
        # is holiday
        res[r_count, 8] = 1 if is_holiday(date_obj) else 0
        # is school holiday
        res[r_count, 9] = 1 if is_school_holiday(date_obj) else 0
        # is exam period
        res[r_count, 10] = 1 if is_exam_period(date_obj) else 0
        # is study year
        res[r_count, 11] = 1 if is_study_year(date_obj) else 0
        # is information day
        res[r_count, 12] = 1 if is_information_day(date_obj) else 0

        # Add date time object to column vector of datetime objects for each cell.
        dt_col[r_count] = date_obj

        # Increment row counter.
        r_count += 1
    return res, dt_col


# is holiday: is the date a work free holiday? ***
def is_holiday(date_obj):
    holidays = [datetime.date(2012, 1, 1), datetime.date(2012, 2, 8),
                datetime.date(2012, 4, 18), datetime.date(2012, 4, 9),
                datetime.date(2012, 4, 27), datetime.date(2012, 5, 1),
                datetime.date(2012, 5, 2), datetime.date(2012, 5, 31),
                datetime.date(2012, 6, 25), datetime.date(2012, 8, 15),
                datetime.date(2012, 10, 31), datetime.date(2012, 11, 1),
                datetime.date(2012, 12, 25), datetime.date(2012, 12, 26)]

    if date_obj.date() in holidays:
        return True
    else:
        return False


# is_school_holiday: os the date during a school holiday? ***
def is_school_holiday(date_obj):
    autumn_break = pd.date_range(pd.datetime(2012, 10, 29), periods=5).tolist()
    christmas_break = pd.date_range(pd.datetime(2012, 12, 24), periods=10).tolist()
    winter_break = pd.date_range(pd.datetime(2012, 2, 20), periods=5).tolist()
    spring_break = pd.date_range(pd.datetime(2012, 4, 30), periods=1).tolist()
    summer_break = pd.date_range(pd.datetime(2012, 6, 22), periods=71).tolist()
    school_holidays = autumn_break + christmas_break + winter_break + spring_break + summer_break

    if date_obj.date() in [x.date() for x in school_holidays]:
        return True
    else:
        return False


# is_exam_period: is the day in the exam period? ***
def is_exam_period(date_obj):
    # Define exam period dates.
    exam_period1 = pd.date_range(pd.datetime(2012, 1, 21), periods=26).tolist()
    exam_period2 = pd.date_range(pd.datetime(2012, 6, 10), periods=26).tolist()
    exam_period3 = pd.date_range(pd.datetime(2012, 9, 2), periods=26).tolist()
    exam_period_days = exam_period1 + exam_period2 + exam_period3

    if date_obj.date() in [x.date() for x in exam_period_days]:
        return True
    else:
        return False


# is_study_year: is the day a working day for students? ***
def is_study_year(date_obj):
    # Define study year dates.
    study_year_days1 = pd.date_range(pd.datetime(2012, 10, 1), periods=90).tolist()
    study_year_days2 = pd.date_range(pd.datetime(2012, 1, 1), periods=191).tolist()
    study_year = study_year_days1 + study_year_days2
    if date_obj.date() in [x.date() for x in study_year]:
        return True
    else:
        return False


# is_study_year: is the day a working day for students? ***
def is_information_day(date_obj):
    # Define information day dates.
    inf_days = pd.date_range(pd.datetime(2012, 2, 10), periods=2).tolist()
    if date_obj.date() in [x.date() for x in inf_days]:
        return True
    else:
        return False


# get_dir_feature: take the route direction column and encode the to-from directions with simple binary encoding.
# Note: the route direction column should come from a matrix for a single line. ***
def get_dir_feature(route_direction_col):
    # Get unique values of column.
    unique_vals = np.unique(route_direction_col)
    # Create a copy of the passed column
    col_cpy = route_direction_col.copy()
    # Start encoding with 0.
    enc = 0
    # Go over unique values of column and encode with numeric values.
    for u in range(len(unique_vals)):
        col_cpy[col_cpy == unique_vals[u]] = str(enc)
        enc += 1

    # Initialize OneHotEncoder instance.
    onehot_encoder = OneHotEncoder(sparse=False)
    # Return one-hot encoded result.
    return onehot_encoder.fit_transform(col_cpy.astype(int).reshape((len(col_cpy), 1)))


# registration_to_num: take registration column and extract the numerical part of each cell.
# Note that some registration numbers contain a star. Also create a star feature and return both in matrix.
# return results as column vector of same size. ***
def registration_to_num(registration_col):
    # Allocate vector for results.
    res = np.empty([len(registration_col), 2], dtype=int)
    # Initialize row counter.
    r_count = 0
    # Go over columns and parse numbers.
    for col in registration_col:
        # If number does not end with a star
        if col[-1] != '*':
            res[r_count, 0] = int(col[7:])
            res[r_count, 1] = 0
        else:
            res[r_count] = int(col[7:-1])
            res[r_count, 1] = 1

        r_count += 1

    # Initialize OneHotEncoder instance.
    onehot_encoder = OneHotEncoder(sparse=False)
    # Return one-hot encoded result.

    # Return result.
    return onehot_encoder.fit_transform(res)


# elpased_time: compute elapsed time in seconds between departure and arrival.
# THIS IS THE TARGET VARIABLE THAT WILL BE PREDICTED ON THE TEST DATA.
# The datetime of arrival will be computed by adding the elapsed time to the start time.
def elapsed_time(start_col, end_col):

    # Define and vectorize functions for parsing dates, for computing delta time
    # and for converting to value in seconds.
    def parse_date(date_str):
        datetime_parse_str = "%Y-%m-%d %H:%M:%S.%f"
        return datetime.datetime.strptime(date_str, datetime_parse_str)

    def time_diff(start, end):
        return end - start

    def to_delta_seconds(timedelta_val):
        return timedelta_val.seconds

    # Get differences between times.
    parse_date_vect = np.vectorize(parse_date)
    time_diff_vect = np.vectorize(time_diff)
    to_delta_seconds_vect = np.vectorize(to_delta_seconds)

    return to_delta_seconds_vect(time_diff_vect(parse_date_vect(start_col), parse_date_vect(end_col)))


# get_features_lpp: get matrix of new features obtained from the datetime and registration columns
# Also append elapsed time feature.
# This function also returns a column vector of datetime objects corresponding to the rows in the datetime_col.
def get_features_lpp(driver_id_col, datetime_col, route_direction_col, registration_col):
    # Get features and stack side by side.
    datetime_features, dt_col = decompose_datetime(datetime_col)
    dir_feature = get_dir_feature(route_direction_col)
    reg_feature = registration_to_num(registration_col)

    # Initialize OneHotEncoder instance.
    onehot_encoder = OneHotEncoder(sparse=False)

    # Stack feature matrices together side by side. Perform one-hot encoding on the driver id.
    res_features = np.hstack((onehot_encoder.fit_transform(driver_id_col.reshape((len(driver_id_col), 1)).astype(float)), datetime_features, dir_feature, reg_feature))
    return res_features, dt_col


# get_features_arso: get matrix of processed weather features stored in csv file with name data_file. ***
# This function also returns a column vector of datetime objects corresponding to the rows in the matrix of weather data.
def get_features_arso(data_file):
    # Read weather data csv file and transform into numpy matrix.
    reader = csv.reader(open(data_file, "r"), delimiter=",")
    x = list(reader)
    # Get weather data matrix.
    weather_data_matrix = np.array(x)

    # Define string for parsing the date from the date column
    datetime_parse_str = "%m/%d/%Y"

    # Allocate column vector for storing the datetime objects.
    dt_col = np.empty((weather_data_matrix.shape[0], 1), dtype=object)
    # Define and initialize row counter.
    r_count = 0
    # Obtain a column of datetime objects corresponding to the rows in the matrix.
    for date in weather_data_matrix[:, 0]:
        # Parse date from date column and add to vector.
        date_obj = datetime.datetime.strptime(date, datetime_parse_str)
        dt_col[r_count] = date_obj
        r_count += 1

    # Return weather data matrix and the datetime column vector.
    # Leave out the date
    return weather_data_matrix[:, 1:], dt_col


# join_lpp_arso: join the lpp and arso feature matrices by the date column.
def join_lpp_arso(features_lpp, features_arso, dt_lpp, dt_arso):

    features_arso_float = features_arso.astype(float)
    # Construct column vector which maps datetime objects in dt_lpp to datetime objects in dt_arso
    mapping = np.empty(len(dt_lpp))
    for (k, dt1) in enumerate(dt_lpp):
        for (l, dt2) in enumerate(dt_arso):
            if dt1[0].date() == dt2[0].date():
                mapping[k] = l

    # Use the mapping to construct a weather features mapping that matches the lpp mapping.
    # Allocate empty matrix for results
    sorted_arso = np.empty([features_lpp.shape[0], features_arso.shape[1]], dtype=float)
    # Go over the mapping vector and build matrix by stacking.
    for (r, m) in enumerate(mapping):
            sorted_arso[r, :] = features_arso_float[int(m), :]

    # Stack feature matrices side by side and return.
    res = np.hstack((features_lpp, sorted_arso))
    return res


# get_feature_matrix_training: get matrix of features obtained from the LPP and ARSO data sheets including the
# values of the target variable obtained from the training set.
def get_feature_matrix_train(lpp_line_matrix):
    # Indices corresponding to columns needed for feature engineering.
    datetime_col = 6
    route_direction_col = 3
    registration_col = 0
    # Get features by performing feature engineering on the LPP data matrix.
    features_lpp, dt_lpp = get_features_lpp(lpp_line_matrix[:, 1], lpp_line_matrix[:, datetime_col],
                                            lpp_line_matrix[:, route_direction_col],
                                            lpp_line_matrix[:, registration_col])
    # Get weather features and datetime linking column vector.
    features_arso, dt_arso = get_features_arso('ljubljana_2012_vreme_raw.csv')
    # Join LPP feature matrix with weather feature matrix.
    res = join_lpp_arso(features_lpp, features_arso, dt_lpp, dt_arso)
    # Get target/dependent variable values.
    target = elapsed_time(lpp_line_matrix[:, 6], lpp_line_matrix[:, 8])
    # Return feature matrix and independent variable column vector.
    return res.astype(float), target


# get_feature_matrix_test: get matrix of features obtained from the LPP and ARSO data sheets without the
# target variable column.
def get_feature_matrix_test(lpp_line_matrix):
    # Indices corresponding to columns needed for feature engineering.
    datetime_col = 6
    route_direction_col = 3
    registration_col = 0
    # Get features by performing feature engineering on the LPP data matrix.
    features_lpp, dt_lpp = get_features_lpp(lpp_line_matrix[:, 1], lpp_line_matrix[:, datetime_col],
                                            lpp_line_matrix[:, route_direction_col],
                                            lpp_line_matrix[:, registration_col])
    # Get weather features and datetime linking column vector.
    features_arso, dt_arso = get_features_arso('ljubljana_2012_vreme_raw.csv')
    # Join LPP feature matrix with weather feature matrix.
    res = join_lpp_arso(features_lpp, features_arso, dt_lpp, dt_arso)
    # return feature matrix.
    return res.astype(float)

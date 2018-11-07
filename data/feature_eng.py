import datetime
import numpy as np


# decompose_datetime: take the date column and create a matrix containing the day, hour, minutes and seconds as columns.
def decompose_datetime(datetime_col):
    # Define string for parsing datetime.
    datetime_parse_str = "%Y-%m-%d %H:%M:%S.%f"

    # Allocate matrix for storing the results.
    res = np.empty([len(datetime_col), 9], dtype=int)

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
        res[r_count, 8] = is_holiday(date_obj)

        # Increment row counter.
        r_count += 1

    return res


def is_holiday(date_obj):
    return None

# get_dir_feature: take the route direction column and encode the to-from directions with simple binary encoding.
def get_dir_feature(route_direction_col):
    # TODO
    pass


# registration_to_num: take registration column and extract the numerical part of each cell.
# return results as column vector of same size.
def registration_to_num(registration_col):
    # TODO
    pass


# append_weather_features: append processed weather features stored in csv file with name data_file.
def append_weather_features(data_file):
    # TODO
    pass
import datetime
import numpy as np
import pandas as pd

# decompose_datetime: take the date column and create a matrix containing the day, hour, minutes and seconds as columns.
def decompose_datetime(datetime_col):
    # Define string for parsing datetime.
    datetime_parse_str = "%Y-%m-%d %H:%M:%S.%f"

    # Allocate matrix for storing the results.
    res = np.empty([len(datetime_col), 14], dtype=int)

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
        # is school holiday
        res[r_count, 10] = 1 if is_exam_period(date_obj) else 0
        # is study year
        res[r_count, 11] = 1 if is_study_year(date_obj) else 0
        # is information day
        res[r_count, 13] = 1 if is_information_day(date_obj) else 0

        # Increment row counter.
        r_count += 1

    return res

# is holiday: is the date a work free holiday?
def is_holiday(date_obj):
    holidays = [datetime.date(2012, 1, 1), datetime.date(2012, 2, 8),
                datetime.date(2012, 4, 18), datetime.date(2012, 4, 9),
                datetime.date(2012, 4, 27), datetime.date(2012, 5, 1),
                datetime.date(2012, 5, 2), datetime.date(2012, 5, 31),
                datetime.date(2012, 6, 25), datetime.date(2012, 8, 15),
                datetime.date(2012, 10, 31), datetime.date(2012, 11, 1),
                datetime.date(2012, 12, 25), datetime.date(2012, 12, 26)]

    if date_obj in holidays:
        return True
    else:
        return False


# is_school_holiday: os the date during a school holiday?
def is_school_holiday(date_obj):
    autumn_break = pd.date_range(pd.datetime(2012, 10, 29), periods=5).tolist()
    christmas_break = pd.date_range(pd.datetime(2012, 12, 24), periods=10).tolist()
    winter_break = pd.date_range(pd.datetime(2012, 2, 20), periods=5).tolist()
    spring_break = pd.date_range(pd.datetime(2012, 4, 30), periods=1).tolist()
    summer_break = pd.date_range(pd.datetime(2012, 6, 22), periods=71).tolist()
    school_holidays = autumn_break + christmas_break + winter_break + spring_break + summer_break

    if date_obj in [x.date() for x in school_holidays]:
        return True
    else:
        return False


# is_exam_period: is the day in the exam period?
def is_exam_period(date_obj):
    # Define exam period dates.
    exam_period1 = pd.date_range(pd.datetime(2012, 1, 21), periods=26).tolist()
    exam_period2 = pd.date_range(pd.datetime(2012, 6, 10), periods=26).tolist()
    exam_period3 = pd.date_range(pd.datetime(2012, 9, 2), periods=26).tolist()
    exam_period_days = exam_period1 + exam_period2 + exam_period3

    if date_obj in [x.date() for x in exam_period_days]:
        return True
    else:
        return False


# is_study_year: is the day a working day for students?
def is_study_year(date_obj):
    # Define study year dates.
    study_year_days1 = pd.date_range(pd.datetime(2012, 10, 1), periods=90).tolist()
    study_year_days2 = pd.date_range(pd.datetime(2012, 1, 1), periods=191).tolist()
    study_year = study_year_days1 + study_year_days2
    if date_obj in [x.date() for x in study_year]:
        return True
    else:
        return False

# is_study_year: is the day a working day for students?
def is_information_day(date_obj):
    # Define information day dates.
    inf_days = pd.date_range(pd.datetime(2012, 2, 10), periods=2).tolist()
    if date_obj in [x.date() for x in inf_days]:
        return True
    else:
        return False


# get_dir_feature: take the route direction column and encode the to-from directions with simple binary encoding.
# Note: the route direction column should come from a matrix for a single line.
def get_dir_feature(route_direction_col):
    # Get unique values of column.
    unique_vals = np.unique(route_direction_col)
    # Start encoding with 0.
    enc = 0
    # Go over unique values of column and encode with numeric values.
    for u in range(len(unique_vals)):
        route_direction_col[route_direction_col == unique_vals[u]] = str(enc)
        enc +=1


# registration_to_num: take registration column and extract the numerical part of each cell.
# return results as column vector of same size.
def registration_to_num(registration_col):
    # Allocate vector for results.
    res = np.empty([len(registration_col), 1], dtype=int)
    # Initialize row counter.
    r_count = 0

    # Go over columns and parse numbers.
    for col in registration_col:
        res[r_count] = int(col[7:])
        
    # Return result.
    return res


# append_weather_features: append processed weather features stored in csv file with name data_file.
def append_weather_features(data_file):
    # TODO
    pass
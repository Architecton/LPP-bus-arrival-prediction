import datetime

########################
# Author: Jernej Vivod #
########################

# get_arrival_time: get predicted time of arrival by adding the predicted trip length
# to the starting time.
def get_arrival_time(start_time, travel_seconds):
    return start_time + datetime.timedelta(seconds=float(travel_seconds))


# get_arrival_time_str: get string representation of the arrival time
# in the correct format.
def get_arrival_time_str(arrival_time):
    return arrival_time[0].strftime('%Y-%m-%d %H:%M:%S.000000\n')


# print_results: print arrival times in specified format.
def print_results(arrival_times, predictions):
    with open('results.txt', 'w') as f:
        for (i_pred, t) in enumerate(arrival_times):
            arrival_time = get_arrival_time(t, predictions[i_pred])
            f.write(get_arrival_time_str(arrival_time))
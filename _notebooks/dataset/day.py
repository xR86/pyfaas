import random

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from typing import *


##### Constants #####

CYCLE_HRS = 24

SUBDIVISION_MINS = 5

PEAKS = ['8:30',
         '19:30']

PEAK_DURATION_MINS = 120

MIN_TRAFFIC_PER_MINUTE = 1
MAX_TRAFFIC_PER_MINUTE = 10

MAX_TRAFFIC_PER_MINUTE_AT_PEAK = 100


#####################

cycle_mins = CYCLE_HRS * 60

time_steps = list(zip(*[(mins // SUBDIVISION_MINS, f'{mins // 60:02d}:{mins % 60:02d}')
                        for mins
                        in range(0, cycle_mins, cycle_mins // 24)]))

time_steps = list(map(list, time_steps))

time_steps[0] += [time_steps[0][-1] + time_steps[0][1]]
time_steps[1] += ['24:00']


def linear_mapping(val: float,
                   lo1: float,
                   hi1: float,
                   lo2: float = 0.0,
                   hi2: float = 1.0) -> float:
    """Linear mapping of one interval [a,b] onto another interval [c,d] for a particular value.

    If x = val in in [a,b] then what is the value of x in [c,d]?

    Args:
        val (float): The value from the first interval to map into the second interval.
        lo1 (float): The lower limit of the first interval. Inclusive.
        hi1 (float): The upper limit of the first interval. Inclusive.
        lo2 (float): The lower limit of the second interval. Inclusive. Defaults to `0.0`.
        hi2 (float): The upper limit of the second interval. Inclusive. Defaults to `1.0`.

    Returns:
        float: The value that `val` from the first interval maps to in the second interval.
    """

    return (val - lo1) * (hi2 - lo2) / (hi1 - lo1) + lo2


def gaussian(x: float,
             peak: float,
             height: float,
             stdev: float) -> float:
    """The y value of a gaussian curve of set height, peak and standard deviation.

    Args:
        x      (float): The x value for which to compute the gaussian curve.
        peak   (float): The position of the peak of the curve.
        height (float): The height of the curve in the peak.
        stdev  (float): The standard deviation of the curve.

    Returns:
        float: The height of the gaussian curve at `x`.
    """

    return height * np.e ** (-1 * (((x - peak) ** 2) / (2 * stdev ** 2)))


def simulate_poll_requests_no_peaks(min_traffic: int,
                                    max_traffic: int,
                                    polling_interval: int) -> int:
    """ Simulates polling the requests for one polling interval of the site. No peaks are considered.

    Args:
        min_traffic      (int): The minimum number of requests per minute.
        max_traffic      (int): The maximum number of requests per minute.
        polling_interval (int): The length of a pooling period (a.k.a. the number of requests will be polled every
                                `polling_interval` minutes). Should be in the interval [0, 1440] (the number of
                                minutes in a day).

    Returns:
        int: The generated number of requests for one polling interval.
    """

    return random.randint(min_traffic * polling_interval, max_traffic * polling_interval)


def simulate_poll_requests_with_peaks(minute: int,
                                      min_traffic: int,
                                      max_traffic: int,
                                      max_traffic_at_peak: int,
                                      peaks: List[int],
                                      peak_duration: int,
                                      polling_interval: int) -> int:
    """Simulates polling the requests for a specific polling interval of the site.

    Args:
        minute              (int)        : The minute of day at which to simulate the traffic. Should be in the interval
                                           [0, 1440] (the number of minutes in a day).
        min_traffic         (int)        : The minimum number of requests per minute during regular traffic times.
        max_traffic         (int)        : The maximum number of requests per minute during regular traffic times.
        max_traffic_at_peak (int)        : The maximum number of requests per minute during peak times.
        peaks               (list of int): The list of times (as minute of day) that are considered "peak times". Each
                                           value should be in the interval [0, 1440] (the number of minutes in a day).
        peak_duration       (int)        : The number of minutes the peak should last.
        polling_interval    (int)        : The length of a pooling period (a.k.a. the number of requests will be polled
                                           every `polling_interval` minutes). Should be in the interval [0, 1440] (the
                                           number of minutes in a day).

    Returns:
        int: The generated number of requests for the specified polling interval.
    """

    return (random.randint(min_traffic * polling_interval, max_traffic * polling_interval) +
            sum(gaussian(minute,
                         peak = peak,
                         height = max_traffic_at_peak * polling_interval,
                         stdev = peak_duration // 4)
                for peak
                in peaks))


def generate_day_dataset(polling_interval: int,
                         min_traffic: int,
                         max_traffic: int,
                         max_traffic_at_peak: int,
                         peak_duration: int,
                         peak_times: List[str],
                         plot: bool = False) -> List[int]:
    """Generate the dataset for a day.

    Args:
        polling_interval    (int)        : The length of a pooling period (a.k.a. the number of requests will be polled
                                           every `polling_interval` minutes). Should be in the interval [0, 1440] (the
                                           number of minutes in a day).
        min_traffic         (int)        : The minimum number of requests per minute during regular traffic times.
        max_traffic         (int)        : The maximum number of requests per minute during regular traffic times.
        max_traffic_at_peak (int)        : The maximum number of requests per minute during peak times.
        peak_duration       (int)        : The number of minutes the peak should last.
        peak_times          (list of str): The list of times (as strings of 24-hour format times, e.g. '08:30', '22:17')
                                           that are considered "peak times". Each value should represent a valid time of
                                           day.
        plot                (bool)       : If this is `True` the returned dataset is also plotted. Mostly used for
                                           debugging and demos.

    Returns:
        list of int: The generated dataset consisting of a list requests per polling interval for each of the polling
                     intervals over the course of a day.
    """

    cycle_mins = 24 * 60
    cycle_time_steps = list(range(0, cycle_mins, polling_interval))

    peaks = [peak.split(':') for peak in peak_times]
    peaks = [int(peak[0]) * 60 + int(peak[1]) for peak in peaks]

    load_with_peaks = [simulate_poll_requests_with_peaks(minute,
                                                         min_traffic = min_traffic,
                                                         max_traffic = max_traffic,
                                                         max_traffic_at_peak = max_traffic_at_peak,
                                                         peaks = peaks,
                                                         peak_duration = peak_duration,
                                                         polling_interval = polling_interval)
                       for minute
                       in cycle_time_steps]

    if plot:

        cycle_mins = CYCLE_HRS * 60

        time_steps = list(zip(*[(mins // SUBDIVISION_MINS, f'{mins // 60:02d}:{mins % 60:02d}')
                                for mins
                                in range(0, cycle_mins, cycle_mins // 24)]))

        time_steps = list(map(list, time_steps))

        time_steps[0] += [time_steps[0][-1] + time_steps[0][1]]
        time_steps[1] += ['24:00']

        plt.plot(load_with_peaks, color = 'b', linewidth = 1)
        plt.title('Traffic with peaks (generate_day_dataset())')
        plt.legend([f'Requests per {polling_interval} minutes (with peaks)'])
        plt.xlabel('Time of day')
        plt.ylabel(f'Requests per {polling_interval} minutes')
        plt.ylim([0, max_traffic_at_peak * polling_interval * 1.25])
        plt.xticks(*time_steps)
        plt.grid(True, linestyle = '--')

        plt.show()

    return load_with_peaks


if __name__ == '__main__':

    PRINT_SAMPLE_SIZE = 5

    print(f'Dataset for cycle of {CYCLE_HRS} hours.')
    print()

    print(f'Number of requests polled every {SUBDIVISION_MINS} minutes.')
    print(f'Peaks at {", ".join(PEAKS[:-1])} and {PEAKS[-1]}.')
    print(f'Each peak lasts approx. {PEAK_DURATION_MINS} minutes.')
    print(f'Max traffic during peak times is {MAX_TRAFFIC_PER_MINUTE_AT_PEAK} per minute.')
    print(f'Usual traffic is between {MIN_TRAFFIC_PER_MINUTE} and {MAX_TRAFFIC_PER_MINUTE} requests per minute.')
    print()

    #####

    cycle_mins = CYCLE_HRS * 60

    cycle_time_steps = list(range(0, cycle_mins, SUBDIVISION_MINS))

    peaks = [peak.split(':') for peak in PEAKS]
    peaks = [int(peak[0]) * 60 + int(peak[1]) for peak in peaks]

    print(f'Cycle lasts {cycle_mins} minutes.')
    print(f'First pollings occur at minutes {cycle_time_steps[:PRINT_SAMPLE_SIZE]}.')
    print(f'Final pollings occur at minutes {cycle_time_steps[-PRINT_SAMPLE_SIZE:]}.')
    print(f'Peaks occur at minutes {peaks}.')
    print()

    #####

    load = [simulate_poll_requests_no_peaks(min_traffic = MIN_TRAFFIC_PER_MINUTE,
                                            max_traffic = MAX_TRAFFIC_PER_MINUTE,
                                            polling_interval = SUBDIVISION_MINS)
            for minute
            in cycle_time_steps]

    print(f'Regular traffic for the first polling intervals is {load[:PRINT_SAMPLE_SIZE]}.')
    print(f'Regular traffic for the final polling intervals is {load[-PRINT_SAMPLE_SIZE:]}.')
    print()

    #####

    load_with_peaks = [simulate_poll_requests_with_peaks(minute,
                                                         min_traffic = MIN_TRAFFIC_PER_MINUTE,
                                                         max_traffic = MAX_TRAFFIC_PER_MINUTE,
                                                         max_traffic_at_peak = MAX_TRAFFIC_PER_MINUTE_AT_PEAK,
                                                         peaks = peaks,
                                                         peak_duration = PEAK_DURATION_MINS,
                                                         polling_interval = SUBDIVISION_MINS)
                       for minute
                       in cycle_time_steps]

    for peak_time, peak in zip(PEAKS, peaks):
        peak_center = peak // SUBDIVISION_MINS
        peak_center = np.round(load_with_peaks[peak_center - PRINT_SAMPLE_SIZE:peak_center + PRINT_SAMPLE_SIZE])
        print(f'Traffic with peaks around peak {peak_time.rjust(5)} is {peak_center}.')
    print()

    #####

    time_steps = list(zip(*[(mins // SUBDIVISION_MINS, f'{mins // 60:02d}:{mins % 60:02d}')
                            for mins
                            in range(0, cycle_mins, cycle_mins // 24)]))

    time_steps = list(map(list, time_steps))

    time_steps[0] += [time_steps[0][-1] + time_steps[0][1]]
    time_steps[1] += ['24:00']

    print(f'Timestamp XX:XX is the end of polling interval XXXX.')
    for polling_interval, timestamp in zip(*time_steps):
        print(' ' * 9, timestamp, ' . ' * 10, polling_interval)
    print()

    #####

    plt.figure(1)

    plt.subplot(2, 1, 1)

    plt.plot(load, color = 'g', linewidth = 1)
    plt.plot(load_with_peaks, color = 'b', linewidth = 1)
    plt.title('Traffic with/without peaks demo')
    plt.legend([f'Requests per {SUBDIVISION_MINS} minutes (without peaks)',
                f'Requests per {SUBDIVISION_MINS} minutes (with peaks)'])
    plt.xlabel('Time of day')
    plt.ylabel(f'Requests per {SUBDIVISION_MINS} minutes')
    plt.ylim([0, MAX_TRAFFIC_PER_MINUTE_AT_PEAK * SUBDIVISION_MINS * 1.25])
    plt.xticks(*time_steps)
    plt.grid(True, linestyle = '--')

    plt.subplot(2, 1, 2)
    plt.plot(generate_day_dataset(polling_interval = SUBDIVISION_MINS,
                                  min_traffic = MIN_TRAFFIC_PER_MINUTE,
                                  max_traffic = MAX_TRAFFIC_PER_MINUTE,
                                  max_traffic_at_peak = MAX_TRAFFIC_PER_MINUTE_AT_PEAK,
                                  peak_duration = PEAK_DURATION_MINS,
                                  peak_times = PEAKS),
             color = 'b',
             linewidth = 1)
    plt.title('Traffic with peaks (generate_day_dataset())')
    plt.legend([f'Requests per {SUBDIVISION_MINS} minutes (with peaks)'])
    plt.xlabel('Time of day')
    plt.ylabel(f'Requests per {SUBDIVISION_MINS} minutes')
    plt.ylim([0, MAX_TRAFFIC_PER_MINUTE_AT_PEAK * SUBDIVISION_MINS * 1.25])
    plt.xticks(*time_steps)
    plt.grid(True, linestyle = '--')

    plt.show()

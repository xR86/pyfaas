import random
import calendar

from typing import *

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from prediction.dataset import generate_day_dataset
from prediction.dataset._consts import *


raise NotImplementedError('Legacy module')


def jitter_month_peak_times(year: int,
                            month: int,
                            peak_times: List[str],
                            variation: int = None) -> List[List[str]]:
    """Extrapolate and add noise to the set of peak times to the length of the specified month of the specified year.

    The noise is added by shifting each peak with a random number of minutes from the interval
    [-`variation`, +`variation`].

    Args:
        year:      (int)        : The year for which to extrapolate.
        month:     (int)        : The month for which to extrapolate.
        peak_times (list of str): The list of times (as strings of 24-hour format times, e.g. '08:30', '22:17') that are
                                  considered "peak times". Each value should represent a valid time of day.
        variation  (int)        : The maximum noise (minutes) that can be added to a peak.

    Returns:
        list of list of str: A list of noisy clones of `peak_times`.
    """

    _, month_length = calendar.monthrange(year, month)

    if variation is None:
        return [peak_times[:] for _ in range(month_length)]

    extrapolated_noisy_peak_times = []
    for _ in range(month_length):
        noisy_peak_times = []
        for peak in peak_times:
            hour, minute = tuple(map(int, peak.split(':')))
            shift = random.randint(-variation, variation)
            minute += shift
            if minute < 0:
                hour -= -minute // 60 + 1
                minute = 60 - (-minute % 60)
            if minute > 59:
                hour += minute // 60
                minute %= 60
            noisy_peak_times += [f'{hour}:{minute}']
        extrapolated_noisy_peak_times += [noisy_peak_times]

    return extrapolated_noisy_peak_times


def generate_month_dataset(year: int,
                           month: int,
                           polling_interval: int,
                           min_traffic: int,
                           max_traffic: int,
                           max_traffic_at_peak: int,
                           peak_durations: List[int],
                           peak_times: List[str],
                           peak_time_max_variation: int = 0,
                           plot: bool = False,
                           weekday_names: List[str] = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'),
                           month_names: List[str] = ('Jan', 'Feb',
                                                     'Mar', 'Apr', 'May',
                                                     'Jun', 'Jul', 'Aug',
                                                     'Sep', 'Oct', 'Nov',
                                                     'Dec')) -> np.ndarray:
    """Generate the dataset for the specified month of the specified year.

    Args:
        year:                   (int)        : The year for which to generate.
        month:                  (int)        : The month for which to generate (1-based).
        polling_interval        (int)        : The length of a pooling period (a.k.a. the number of requests will be
                                               polled every `polling_interval` minutes). Should be in the interval
                                               [0, 1440] (the number of minutes in a day).
        min_traffic             (int)        : The minimum number of requests per minute during regular times.
        max_traffic             (int)        : The maximum number of requests per minute during regular times.
        max_traffic_at_peak     (int)        : The maximum number of requests per minute during peak times.
        peak_durations          (list of int): The list of numbers of minutes each peak should last.
        peak_times              (list of str): The list of times (as strings of 24-hour format times, e.g. '08:30',
                                               '22:17') that are considered "peak times". Each value should represent a
                                               valid time of day.
        peak_time_max_variation (int)        : The maximum time interval, in minutes, that the peaks can be shifted
                                               with  when adding noise. Defaults to `0` (no shifting of peaks).
        plot                    (bool)       : If this is `True` the returned dataset is also plotted. Mostly used for
                                               debugging and demos. Defaults to `False`.
        weekday_names           (list of str): The names of weekdays that will be used for plotting if `plot` is
                                               `True`. Defaults to the English shortened weekday names
                                               (Mon, Tue, etc.).
        month_names             (list of str): The names of months that will be used for plotting is `plot` is `True`.
                                               Defaults to the English shortened month names (Jan, Feb, etc.).

    Returns:
        np.ndarray int: A 2d Numpy array of datasets, each row representing the traffic for one day as returned by
                        `dataset.day.generate_day_dataset`.
    """

    month_start_weekday, month_length = calendar.monthrange(year, month)

    noisy_peak_times = jitter_month_peak_times(year = year,
                                               month = month,
                                               peak_times = peak_times,
                                               variation = peak_time_max_variation)

    month_day_traffic = np.array([generate_day_dataset(polling_interval = polling_interval,
                                                       min_traffic = min_traffic,
                                                       max_traffic = max_traffic,
                                                       max_traffic_at_peak = max_traffic_at_peak,
                                                       peak_durations = peak_durations,
                                                       peak_times = noisy_peak_times_for_day)
                                  for i, noisy_peak_times_for_day
                                  in zip(range(month_length), noisy_peak_times)])

    if plot:

        cycle_mins = 24 * 60

        time_steps = list(zip(*[(mins // polling_interval, f'{mins // 60:02d}:{mins % 60:02d}')
                                for mins
                                in range(0, cycle_mins, cycle_mins // 24)]))

        time_steps = list(map(list, time_steps))

        time_steps[0] += [time_steps[0][-1] + time_steps[0][1]]
        time_steps[1] += ['24:00']

        time_stamps = time_steps[1][::12]
        time_steps = time_steps[0][::12]

        plt.figure(1)
        plt.subplots_adjust(wspace = 0.35, hspace = 0.35)

        plt.suptitle(f'{month_names[month - 1]}, {year}', fontsize = 20, fontweight = 'bold')

        for i, weekday_name in enumerate(weekday_names):
            plt.subplot(5, 7, i + 1)
            plt.title(weekday_name, fontsize = 15, fontweight = 'bold')
            if i < month_start_weekday:
                plt.yticks([], [])
                plt.xticks([], [])

        for day_ind, day in enumerate(month_day_traffic):
            plt.subplot(5, 7, month_start_weekday + day_ind + 1)
            plt.plot(day, color = 'b', linewidth = 0.5)
            plt.ylim([0, max_traffic_at_peak * polling_interval * 1.25])
            plt.xticks(time_steps, time_stamps, rotation = 0)
            plt.grid(True, linestyle = '--')
            day_suff = 'st' if day_ind % 10 == 0 else 'nd' if day_ind % 10 == 1 else 'rd' if day_ind % 10 == 2 else 'th'
            plt.text(time_steps[-1] * 0.8,
                     max_traffic_at_peak * polling_interval * 1.1,
                     f'{day_ind + 1}{day_suff}',
                     fontweight = 'bold')

        for i in range(len(month_day_traffic) + month_start_weekday, 5 * 7):
            plt.subplot(5, 7, i + 1)
            plt.yticks([], [])
            plt.xticks([], [])

        plt.show()

    return month_day_traffic


if __name__ == '__main__':
    generate_month_dataset(year = 2018,
                           month = 2,
                           polling_interval = POLLING_INTERVAL,
                           min_traffic = MIN_TRAFFIC_PER_MINUTE,
                           max_traffic = MAX_TRAFFIC_PER_MINUTE,
                           max_traffic_at_peak = MAX_TRAFFIC_PER_MINUTE_AT_PEAK,
                           peak_durations = PEAK_DURATIONS_MINS,
                           peak_times = PEAKS,
                           peak_time_max_variation = PEAK_MAX_VARIATION_MINS,
                           plot = True)

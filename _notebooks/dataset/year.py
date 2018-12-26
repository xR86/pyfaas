import itertools

from typing import *

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

import dataset.day
import dataset.month


def floor_avg(iterable: Sequence[int]) -> int:
    return sum(iterable) // len(iterable)


def avg_traffic_per_day(days: List[List[int]]) -> List[int]:
    days_t = np.array(days).transpose()
    return [floor_avg(time_step) for time_step in days_t]


def generate_year_dataset(year: int,
                          polling_interval: int,
                          min_traffic: int,
                          max_traffic: int,
                          max_traffic_at_peak: int,
                          peak_duration: int,
                          peak_times: List[str],
                          peak_time_max_variation: int = 0,
                          plot: bool = False,
                          month_names: List[str] = ('Jan', 'Feb',
                                                    'Mar', 'Apr', 'May',
                                                    'Jun', 'Jul', 'Aug',
                                                    'Sep', 'Oct', 'Nov',
                                                    'Dec')) -> List[List[int]]:
    """Generate noisy data for one year using the month data generation function `dataset.month.generate_month_dataset`.

    Args:
        year:                   (int)         : The year for which to generate.
        polling_interval        (int)         : The length of a pooling period (a.k.a. the number of requests will be
                                                polled every `polling_interval` minutes). Should be in the interval
                                                [0, 1440] (the number of minutes in a day).
        min_traffic             (int)         : The minimum number of requests per minute during regular times.
        max_traffic             (int)         : The maximum number of requests per minute during regular times.
        max_traffic_at_peak     (int)         : The maximum number of requests per minute during peak times.
        peak_duration           (int)         : The number of minutes the peak should last.
        peak_times              (list of str) : The list of times (as strings of 24-hour format times, e.g. '08:30',
                                                '22:17') that are considered "peak times". Each value should represent a
                                                valid time of day.
        peak_time_max_variation (int)         : The maximum time interval, in minutes, that the peaks can be shifted
                                                with  when adding noise. Defaults to `0` (no shifting of peaks).
        plot                    (bool)        : If this is `True` the returned dataset is also plotted. Mostly used for
                                                debugging and demos. Defaults to `False`.
        month_names             (list of str) : The names of months that will be used for plotting is `plot` is `True`.
                                                Defaults to the English shortened month names (Jan, Feb, etc.).

    Returns:
        list of list of int: A list of datasets, each representing the traffic for one day as returned by
                             `dataset.day.generate_month_dataset`.
    """

    year_month_traffic = []
    for month_ind, month_name in enumerate(month_names):
        year_month_traffic += [dataset.month.generate_month_dataset(year = year,
                                                                    month = month_ind + 1,
                                                                    polling_interval = polling_interval,
                                                                    min_traffic = min_traffic,
                                                                    max_traffic = max_traffic,
                                                                    max_traffic_at_peak = max_traffic_at_peak,
                                                                    peak_duration = peak_duration,
                                                                    peak_times = peak_times,
                                                                    peak_time_max_variation = peak_time_max_variation,
                                                                    plot = False)]

    if plot:

        year_month_avg_traffic = [avg_traffic_per_day(month_traffic) for month_traffic in year_month_traffic]
        max_traffic = max(max(month_avg_traffic) for month_avg_traffic in year_month_avg_traffic)

        time_steps = dataset.day.time_steps[0][::12]
        time_stamps = dataset.day.time_steps[1][::12]

        plt.figure(1)
        plt.subplots_adjust(hspace = 0.35)

        plt.suptitle(f'Average traffic per day for {year}', fontsize = 20, fontweight = 'bold')

        for month_ind, month_name in enumerate(month_names):
            plt.subplot(4, 3, month_ind + 1)
            plt.title(month_name, fontsize = 15, fontweight = 'bold')
            plt.plot(year_month_avg_traffic[month_ind], color = 'b', linewidth = 0.5)
            plt.ylim([0, max_traffic * 1.25])
            plt.xticks(time_steps, time_stamps, rotation = 0)
            plt.grid(True, linestyle = '--')

        plt.show()

    year_day_traffic = list(itertools.chain.from_iterable(year_month_traffic))

    return year_day_traffic


if __name__ == '__main__':
    year = 2018
    generate_year_dataset(year = year,
                          polling_interval = 5,
                          min_traffic = 1,
                          max_traffic = 50,
                          max_traffic_at_peak = 100,
                          peak_times = dataset.day.PEAKS,
                          peak_duration = 120,
                          peak_time_max_variation = 120,
                          plot = True)
import random
import calendar
import datetime

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.backends.backend_pdf
from matplotlib import pyplot as plt

from typing import *

from prediction.dataset import generate_day_dataset


raise NotImplementedError('Legacy module')


def jitter_peak_times(peak_times: List[str],
                      variation: int,
                      days: int) -> List[List[str]]:
    """Clone and jitter and the set of peak times for the specified number of days.

    The noise is added by shifting each peak with a random number of minutes from the interval
    [-`variation`, +`variation`].

    Args:
        peak_times (list of str): The list of times (as strings of 24-hour format times, e.g. '08:30', '22:17') that are
                                  considered "peak times". Each value should represent a valid time of day.
        variation  (int)        : The maximum noise (minutes) that can be added to a peak.
        days       (int)        : The number of days to extrapolate.

    Returns:
        list of list of str: A list of noisy clones of `peak_times`.
    """

    if variation is None:
        return [peak_times[:] for _ in range(days)]

    jittered_range_peak_times = []
    for _ in range(days):
        jittered_peak_times = []
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
            jittered_peak_times += [f'{hour}:{minute}']
        jittered_range_peak_times += [jittered_peak_times]

    return jittered_range_peak_times


def jitter_peak_durations(peak_durations: List[int],
                          variation: int,
                          days: int) -> List[List[int]]:
    """Clone and jitter the set of peak durations for the specified number of days.

    The noise is added by shifting each duration with a random number of minutes from the interval
    [-`variation`, +`variation`].

    Args:
        peak_durations (list of int): The list of durations of each peak.
        variation  (int)        : The maximum noise (minutes) that can be added to a peak.
        days       (int)        : The number of days to extrapolate.

    Returns:
        list of list of str: A list of noisy clones of `peak_durations`.
    """

    if variation is None:
        return [peak_durations[:] for _ in range(days)]

    jittered_range_peak_durations = []
    for _ in range(days):
        jittered_peak_durations = [max(0, peak_duration + random.randint(-variation, variation))
                                   for peak_duration
                                   in peak_durations]
        jittered_range_peak_durations += [jittered_peak_durations]

    return jittered_range_peak_durations


def generate_range_dataset(start_date: datetime.date,
                           end_date: datetime.date,
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
    """Generate noisy data for one year using the month data generation function `dataset.month.generate_month_dataset`.

    Args:
        start_date              (datetime.date): The start date of the generated dataset, inclusive.
        end_date                (datetime.date): The end date of the generated dataset, inclusive.
        polling_interval        (int)          : The length of a pooling period (a.k.a. the number of requests will be
                                                 polled every `polling_interval` minutes). Should be in the interval
                                                 [0, 1440] (the number of minutes in a day).
        min_traffic             (int)          : The minimum number of requests per minute during regular times.
        max_traffic             (int)          : The maximum number of requests per minute during regular times.
        max_traffic_at_peak     (int)          : The maximum number of requests per minute during peak times.
        peak_durations          (list of int)  : The list of numbers of minutes each peak should last.
        peak_times              (list of str)  : The list of times (as strings of 24-hour format times, e.g. '08:30',
                                                 '22:17') that are considered "peak times". Each value should represent
                                                 a valid time of day.
        peak_time_max_variation (int)          : The maximum time interval, in minutes, that the peaks can be shifted
                                                 with  when adding noise. Defaults to `0` (no shifting of peaks).
        plot                    (bool)         : If this is `True` the returned dataset is also plotted. Mostly used for
                                                 debugging and demos. Defaults to `False`.
        weekday_names           (list of str)  : The names of weekdays that will be used for plotting if `plot` is
                                                 `True`. Defaults to the English shortened weekday names
                                                 (Mon, Tue, etc.).
        month_names             (list of str)  : The names of months that will be used for plotting is `plot` is `True`.
                                                 Defaults to the English shortened month names (Jan, Feb, etc.).

    Returns:
        np.ndarray of int: A 2d Numpy array of datasets, each row representing the traffic for one day as returned by
                          `dataset.day.generate_month_dataset`.
    """

    if start_date >= end_date:
        raise ValueError(f'Stop date ({end_date}) must be after the start date ({start_date}).')

    range_days = (end_date - start_date).days

    jittered_peaks = jitter_peak_times(peak_times, peak_time_max_variation, range_days)


    traffic = np.array([generate_day_dataset(polling_interval = polling_interval,
                                             min_traffic = min_traffic,
                                             max_traffic = max_traffic,
                                             max_traffic_at_peak = max_traffic_at_peak,
                                             peak_durations = peak_durations,
                                             peak_times = jittered_day_peaks,
                                             plot = False)
                        for day_ind, jittered_day_peaks
                        in zip(range(range_days), jittered_peaks)])

    if plot:

        raise NotImplementedError

        # Months in range.

        months_left_in_start_year = 12 - start_date.month + 1
        months_passed_in_stop_year = end_date.month
        months_in_complete_years = (end_date.year - start_date.year - 1) * 12
        months = months_left_in_start_year + months_in_complete_years + months_passed_in_stop_year

        # Timestamps for daily traffic.

        cycle_mins = 24 * 60

        time_steps = list(zip(*[(mins // polling_interval, f'{mins // 60:02d}:{mins % 60:02d}')
                                for mins
                                in range(0, cycle_mins, cycle_mins // 24)]))

        time_steps = list(map(list, time_steps))

        time_steps[0] += [time_steps[0][-1] + time_steps[0][1]]
        time_steps[1] += ['24:00']

        time_stamps = time_steps[1][::12]
        time_steps = time_steps[0][::12]

        # Day offset for month.

        offset = calendar.monthrange(start_date.year, start_date.month)[1] - start_date.day

        # Create pdf.

        pdf = matplotlib.backends.backend_pdf.PdfPages('output.pdf')

        for month_ind in range(1, (months-1) + 1):

            year = start_date.year + (start_date.month - 1 + month_ind) // 12
            month = (start_date.month - 1 + month_ind) % 12 + 1
            month_start_weekday, month_length = calendar.monthrange(year, month)

            offset += month_length

            fig = plt.figure(month)
            fig.subplots_adjust(wspace = 0.35, hspace = 0.35)

            fig.suptitle(f'{month_names[month - 1]}, {year}', fontsize = 20, fontweight = 'bold')

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
            pdf.savefig(fig)


        pdf.close()


        plt.show()

    return traffic


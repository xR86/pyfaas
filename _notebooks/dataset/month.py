import calendar

from typing import *

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import dataset.day


def generate_month_dataset(year: int,
                           month: int,
                           polling_interval: int,
                           min_traffic: int,
                           max_traffic: int,
                           max_traffic_at_peak: int,
                           peak_duration: int,
                           peak_times: List[str],
                           plot: bool = False) -> List[List[int]]:

    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    month_start_weekday, month_length = calendar.monthrange(year, month)

    month_day_traffic = [dataset.day.generate_day_dataset(polling_interval = polling_interval,
                                                          min_traffic = min_traffic,
                                                          max_traffic = max_traffic,
                                                          max_traffic_at_peak = max_traffic_at_peak,
                                                          peak_duration = peak_duration,
                                                          peak_times = peak_times)
                         for _
                         in range(month_length)]

    if plot:

        time_steps = dataset.day.time_steps[0][::12]
        time_stamps = dataset.day.time_steps[1][::12]

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
            plt.ylim([0, dataset.day.MAX_TRAFFIC_PER_MINUTE_AT_PEAK * dataset.day.SUBDIVISION_MINS * 1.25])
            plt.xticks(time_steps, time_stamps, rotation = 0)
            plt.grid(True, linestyle = '--')
            day_suff = 'st' if day_ind % 10 == 0 else 'nd' if day_ind % 10 == 1 else 'rd' if day_ind % 10 == 2 else 'th'
            plt.text(time_steps[-1] * 0.8,
                     dataset.day.MAX_TRAFFIC_PER_MINUTE_AT_PEAK * dataset.day.SUBDIVISION_MINS * 1.1,
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
                           polling_interval = 5,
                           min_traffic = 1,
                           max_traffic = 10,
                           max_traffic_at_peak = 100,
                           peak_duration = 120,
                           peak_times = dataset.day.PEAKS,
                           plot = True)

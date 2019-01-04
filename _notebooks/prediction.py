import sys
import time as t
import random
import calendar
from datetime import datetime
from datetime import timedelta
from functools import partial

import numpy as np

from sklearn.linear_model import LinearRegression

from prediction.dataset import generate_day_dataset


from typing import *


WEEKDAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')

START_DATE = (2014, 3, 25)
END_DATE = (2014, 3, 27)


time = partial(datetime, 1, 1, 2)


class Dataset:

    _start_date: datetime
    _end_date: datetime

    _polling_interval: int

    _min_traffic: int
    _max_traffic: int

    _peak_times: List[datetime]
    _peak_time_max_variation: int

    _peak_durations: List[int]
    _peak_duration_max_variation: int

    _max_traffic_at_peaks: List[int]
    _max_traffic_at_peak_variation: int

    _data: List[np.ndarray]

    def __init__(self,

                 start_date: Tuple[int, int, int],
                 end_date: Tuple[int, int, int],

                 polling_interval: int,

                 min_traffic: int,
                 max_traffic: int,

                 peak_times: List[Tuple[int, int]],
                 peak_time_max_variation: int,

                 peak_durations: List[int],
                 peak_duration_max_variation: int,

                 max_traffic_at_peaks: List[int],
                 max_traffic_at_peak_variation: int):
        """Generate noisy data for the time period between 2 given dates `dataset._day.generate_day_dataset`.

            Args:
                start_date                    (3-tuple of ints)        : The start date of the generated dataset,
                                                                         inclusive. Must be a valid 3-tuple representing
                                                                         a (year, month, day) triple.
                end_date                      (3-tuple of ints)        : The end date of the generated dataset,
                                                                         exclusive. Must be a valid 3-tuple representing
                                                                         a (year, month, day) triple.
                polling_interval              (int)                    : The length of a pooling period (a.k.a. the
                                                                         number of requests will be polled every
                                                                         `polling_interval` minutes). Should be in the
                                                                         interval [0, 1440] (the number of minutes in a
                                                                         day) and a divisor of 1440.
                min_traffic                   (int)                    : The minimum number of requests per minute
                                                                         during regular times. Must be positive.
                max_traffic                   (int)                    : The maximum number of requests per minute
                                                                         during regular times. Must be positive.
                peak_times                    (list of 2-tuple of ints): The list of times that are considered
                                                                         "peak times". Each value should represent a
                                                                         valid (hour, minute) pair.
                peak_time_max_variation       (int)                    : The maximum time interval, in minutes, that the
                                                                         peaks can be jittered. Must be in the
                                                                         interval [0, 1440].
                peak_durations                (list of int)            : The list of numbers of minutes each peak should
                                                                         last. Should have the same length as
                                                                         `peak_times` and contain positive numbers.
                peak_duration_max_variation   (int)                    : The maximum number of minutes that the peak
                                                                         durations can be jittered with. Must be in the
                                                                         interval [0, 1440].
                max_traffic_at_peaks          (int)                    : The list of maximum numbers of requests per
                                                                         minute during each peak time. Should have the
                                                                         same length as `peak_times`.
                max_traffic_at_peak_variation (int)                    : The maximum number of requests that the maximum
                                                                         traffic during peaks can be jittered with. Must
                                                                         be positive.

            Raises:
                ValueError: In case any of the requirement for the arguments are not respected.
        """

        # Initializing data generation parameters.

        self._initialize_start_and_end_date(start_date = start_date,
                                            end_date = end_date)

        self._initialize_polling_interval(polling_interval = polling_interval)

        self._initialize_min_and_max_traffic(min_traffic = min_traffic,
                                             max_traffic = max_traffic)

        self._initialize_peak_times(peak_times = peak_times,
                                    peak_time_max_variation = peak_time_max_variation)

        self._initialize_peak_durations(peak_durations = peak_durations,
                                        peak_duration_max_variation = peak_duration_max_variation)

        self._initialize_max_traffic_at_peaks(max_traffic_at_peaks = max_traffic_at_peaks,
                                              max_traffic_at_peak_variation = max_traffic_at_peak_variation)

        self._generate_data()

    def _initialize_start_and_end_date(self,
                                       start_date: Tuple[int, int, int],
                                       end_date: Tuple[int, int, int]):

        start_date = datetime(*start_date)
        end_date = datetime(*end_date)

        if start_date >= end_date:
            raise ValueError('The stop date must be after the start date.')

        self._start_date = start_date
        self._end_date = end_date

    def _initialize_polling_interval(self, polling_interval: int):

        if polling_interval <= 0:
            raise ValueError('The polling interval must be positive.')

        if 1440 % polling_interval != 0:
            raise ValueError('The polling interval must be a divisor of 1440 (the number of minutes in a day).')

        self._polling_interval = polling_interval

    def _initialize_min_and_max_traffic(self, min_traffic: int, max_traffic: int):

        if min_traffic <= 0:
            raise ValueError('The minimum traffic must be positive.')

        if max_traffic <= 0:
            raise ValueError('The maximum traffic must be positive.')

        if max_traffic < min_traffic:
            raise ValueError('The maximum traffic must be smaller than the minimum traffic.')

        self._min_traffic = min_traffic
        self._max_traffic = max_traffic

    def _initialize_peak_times(self,
                               peak_times: List[Tuple[int, int]],
                               peak_time_max_variation: int):

        self._peak_times = [time(hour = peak_time[0], minute = peak_time[1])
                            for peak_time
                            in peak_times]

        if peak_time_max_variation <= 0:
            raise ValueError('The maximum peak time variation must be positive.')

        if peak_time_max_variation >= 1440:
            raise ValueError('The maximum peak time variation must be less than one day.')

        self._peak_time_max_variation = peak_time_max_variation

    def _initialize_peak_durations(self,
                                   peak_durations: List[int],
                                   peak_duration_max_variation: int):

        if len(peak_durations) != len(self.peak_times):
            raise ValueError('The peak duration list must have the same length as the peak times list.')

        if any(peak_duration <= 0 for peak_duration in peak_durations):
            raise ValueError('The peak durations must be positive.')

        self._peak_durations = peak_durations

        if peak_duration_max_variation <= 0:
            raise ValueError('The maximum peak duration variation must be positive.')

        if peak_duration_max_variation >= 1440:
            raise ValueError('The maximum peak duration variation must be less than one day.')

        self._peak_duration_max_variation = peak_duration_max_variation

    def _initialize_max_traffic_at_peaks(self,
                                         max_traffic_at_peaks: List[int],
                                         max_traffic_at_peak_variation: int):

        if len(max_traffic_at_peaks) != len(self.peak_times):
            raise ValueError('The maximum traffic at peaks list must have the same length as the peak times list.')

        self._max_traffic_at_peaks = max_traffic_at_peaks

        if max_traffic_at_peak_variation <= 0:
            raise ValueError('The maximum peak traffic variation must be positive.')

        self._max_traffic_at_peak_variation = max_traffic_at_peak_variation

    def _jitter_peak_times(self) -> List[List[datetime]]:

        jittered_peak_times_for_range = []
        for day_ind in range(self.days):

            jittered_peak_times = []
            for peak in self.peak_times:

                shift_mins = random.randint(-self.peak_time_max_variation, self.peak_time_max_variation)
                shift_seconds = shift_mins * 60
                shift = timedelta(seconds = shift_seconds)

                jittered_peak = peak + shift

                # Treat underflow and overflow

                if jittered_peak.day < peak.day:
                    jittered_peak = time(hour = 0, minute = 0)

                if jittered_peak.day > peak.day:
                    jittered_peak = time(hour = 23, minute = 59)

                jittered_peak_times += [jittered_peak]

            jittered_peak_times_for_range += [jittered_peak_times]

        return jittered_peak_times_for_range

    def _jitter_peak_durations(self) -> List[List[int]]:

        jittered_peak_durations_for_range = []
        for day_ind in range(self.days):

            jittered_peak_durations = []
            for peak_duration in self.peak_durations:

                shift = random.randint(-self.peak_duration_max_variation, self.peak_duration_max_variation)
                jittered_peak_duration = np.asscalar(np.clip(peak_duration + shift, 1, 1439))

                jittered_peak_durations += [jittered_peak_duration]

            jittered_peak_durations_for_range += [jittered_peak_durations]

        return jittered_peak_durations_for_range

    def _jitter_max_traffic_at_peaks(self) -> List[List[int]]:

        jittered_max_traffic_at_peaks_for_range = []
        for day_ind in range(self.days):

            jittered_max_traffic_at_peaks = []
            for max_traffic_at_peak in self.max_traffic_at_peaks:

                shift = random.randint(-self.max_traffic_at_peak_variation, self.max_traffic_at_peak_variation)
                jittered_max_traffic_at_peak = max_traffic_at_peak + shift

                jittered_max_traffic_at_peaks += [jittered_max_traffic_at_peak]

            jittered_max_traffic_at_peaks_for_range += [jittered_max_traffic_at_peaks]

        return jittered_max_traffic_at_peaks_for_range

    def _generate_data(self):

        jittered_peak_times_for_range = self._jitter_peak_times()
        jittered_peak_durations_for_range = self._jitter_peak_durations()
        jittered_max_traffic_at_peaks_for_range = self._jitter_max_traffic_at_peaks()

        self._data = [generate_day_dataset(polling_interval = self.polling_interval,
                                           min_traffic = self.min_traffic,
                                           max_traffic = self.max_traffic,
                                           peak_times = jittered_peak_times,
                                           peak_durations = jittered_peak_durations,
                                           max_traffic_at_peaks = jittered_max_traffic_at_peaks)
                      for jittered_peak_times, jittered_peak_durations, jittered_max_traffic_at_peaks
                      in zip(jittered_peak_times_for_range,
                             jittered_peak_durations_for_range,
                             jittered_max_traffic_at_peaks_for_range)]

    def _expand_day(self, day: np.ndarray) -> List[Tuple[int, int, int]]:
        result = []
        curr_time = [0, 0]
        for moment in day:
            if curr_time[1] >= 60:
                curr_time[0] += 1
                curr_time[1] %= 60
            result += [(*curr_time, moment)]
            curr_time[1] += self.polling_interval
        return result

    @property
    def start_date(self) -> datetime:
        return self._start_date

    @property
    def end_date(self) -> datetime:
        return self._end_date

    @property
    def polling_interval(self) -> int:
        return self._polling_interval

    @property
    def min_traffic(self) -> int:
        return self._min_traffic

    @property
    def max_traffic(self) -> int:
        return self._max_traffic

    @property
    def peak_times(self) -> List[datetime]:
        return self._peak_times

    @property
    def peak_time_max_variation(self) -> int:
        return self._peak_time_max_variation

    @property
    def peak_durations(self) -> List[int]:
        return self._peak_durations

    @property
    def peak_duration_max_variation(self) -> int:
        return self._peak_duration_max_variation

    @property
    def max_traffic_at_peaks(self) -> List[int]:
        return self._max_traffic_at_peaks

    @property
    def max_traffic_at_peak_variation(self) -> int:
        return self._max_traffic_at_peak_variation

    @property
    def days(self):
        return (self.end_date - self.start_date).days

    @property
    def data(self):
        return self._data

    def as_days(self) -> np.ndarray:
        data = []
        for day in self.data:
            hr_min_loads = self._expand_day(day)
            data.extend(hr_min_loads)
        data = np.array(data).round()
        return data

    def as_weeks(self) -> np.ndarray:
        data = []
        curr_date = self.start_date
        delta = timedelta(days = 1)
        for day_ind, day in enumerate(self.data):
            wkday = curr_date.weekday()
            hr_min_loads = self._expand_day(day)
            wkday_hr_min_loads = [(wkday, hr, minute, load)
                                  for hr, minute, load
                                  in hr_min_loads]
            data.extend(wkday_hr_min_loads)
            curr_date += delta
        data = np.array(data).round()
        return data

    def as_months(self) -> np.ndarray:
        data = []
        curr_date = self.start_date
        delta = timedelta(days = 1)
        for day_ind, day in enumerate(self.data):
            mthday = curr_date.day
            hr_min_loads = self._expand_day(day)
            mthday_hr_min_loads = [(mthday, hr, minute, load)
                                  for hr, minute, load
                                  in hr_min_loads]
            data.extend(mthday_hr_min_loads)
            curr_date += delta
        data = np.array(data).round()
        return data

    def as_years(self) -> np.ndarray:
        data = []
        curr_date = self.start_date
        delta = timedelta(days = 1)
        for day_ind, day in enumerate(self.data):
            mth = curr_date.month
            mthday = curr_date.day
            hr_min_loads = self._expand_day(day)
            mnday_hr_min_loads = [(mth, mthday, hr, minute, load)
                                  for hr, minute, load
                                  in hr_min_loads]
            data.extend(mnday_hr_min_loads)
            curr_date += delta
        data = np.array(data).round()
        return data


if __name__ == '__main__':
    data = Dataset(start_date = (2014, 9, 15),
                   end_date = (2025, 5, 29),

                   polling_interval = 5,

                   min_traffic = 5,
                   max_traffic = 15,

                   peak_times = [(8, 30), (19, 30)],
                   peak_time_max_variation = 30,

                   peak_durations = [60, 120],
                   peak_duration_max_variation = 60,

                   max_traffic_at_peaks = [75, 150],
                   max_traffic_at_peak_variation = 50)

    start = t.clock()
    d = data.as_weeks()
    stop = t.clock()
    print(f'Took {stop-start:.2f} s')

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.plot(d[:, 0] * 24 * 60 + d[:, 1] * 60 + d[:, 2], d[:, 3],
             marker = '.',
             linestyle = '')
    plt.xticks(np.arange(0, 1440 * 7 + 100, 1440))
    plt.grid(True)
    plt.show()





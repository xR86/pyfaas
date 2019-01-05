import sys
import time as t
import random
import calendar
from datetime import datetime
from datetime import timedelta
from functools import partial

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    _weekend: bool
    _min_weekend_traffic_variation: int
    _max_weekend_traffic_variation: int

    _black_friday: bool
    _min_black_friday_traffic_variation: int
    _max_black_friday_traffic_variation: int

    _holiday_season: bool
    _min_holiday_season_traffic_variation: int
    _max_holiday_season_traffic_variation: int

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
                 max_traffic_at_peak_variation: int,

                 weekend: bool = False,
                 min_weekend_traffic_variation: int = 0,
                 max_weekend_traffic_variation: int = 0,

                 black_friday: bool = False,
                 min_black_friday_traffic_variation: int = 0,
                 max_black_friday_traffic_variation: int = 0,

                 holiday_season: bool = False,
                 min_holiday_season_traffic_variation: int = 0,
                 max_holiday_season_traffic_variation: int = 0):
        """Generate noisy data for the time period between 2 given dates `dataset._day.generate_day_dataset`.

            Args:
                start_date (3-tuple of int): The start date of the generated dataset, inclusive. Must be a valid 3-tuple
                                             representing a (year, month, day) triple.
                end_date   (3-tuple of int): The end date of the generated dataset, exclusive. Must be a valid 3-tuple
                                             representing a (year, month, day) triple.

                polling_interval (int): The length of a pooling period (a.k.a. the number of requests will be polled
                                        every `polling_interval` minutes). Should be positive and a divisor of 1440 (the
                                        number of minutes in a day).

                min_traffic (int): The minimum number of requests per minute during regular times. Must be positive.
                max_traffic (int): The maximum number of requests per minute during regular times. Must be positive and
                                   higher than `min_traffic`.

                peak_times (list of 2-tuple of int): The list of times that are considered "peak times".  Each value
                                                     should represent a valid (hour, minute) pair.
                peak_time_max_variation (int): The maximum time interval, in minutes, that the peaks can be jittered.
                                               Must be in the interval [0, 1440].

                peak_durations (list of int): The list of numbers of minutes each peak should last. Should have the same
                                              length as `peak_times` and contain positive numbers.
                peak_duration_max_variation (int): The maximum number of minutes that the peak durations can be jittered
                                                   with. Must be in the interval [0, 1440].

                max_traffic_at_peaks (int): The list of maximum numbers of requests per minute during each peak time.
                                            Should have the same length as `peak_times`.
                max_traffic_at_peak_variation (int): The maximum number of requests that the maximum traffic during
                                                     peaks can be jittered with. Must be positive.

                weekend (bool): Whether to simulate increased traffic during the weekend.
                min_weekend_traffic_variation (int): The minimum number of requests that the regular traffic during
                                                     weekends can be jittered with.
                min_weekend_traffic_variation (int): The maximum number of requests that the regular traffic during
                                                     weekends can be jittered with. Must be higher or equal to
                                                     `min_weekend_traffic_variation`.

                black_friday (bool): Whether to simulate a heavily increased traffic on black fridays (29 Nov).
                min_black_friday_traffic_variation (int): The minimum number of requests that the regular traffic during
                                                          black friday can be jittered with. Must be positive.
                max_black_friday_traffic_variation (int): The maximum number of requests that the regular traffic during
                                                          black friday can be jittered with. Must be positive and higher
                                                          or equal to `min_black_friday_traffic_variation`.

                holiday_season (bool): Whether to simulate an increased traffic during the holiday season (Dec) and
                                       weekend-like during mandated free days (24/25/31 Dec and 1/2 Jan).
                min_holiday_season_traffic_variation (int): The minimum number of requests that the regular traffic
                                                            during holiday season can be jittered with. Must be
                                                            positive.
                max_holiday_season_traffic_variation (int): The maximum number of requests that the regular traffic
                                                            during holiday season can be jittered with. Must be positive
                                                            and higher or equal to
                                                            `min_holiday_season_traffic_variation`.

            Raises:
                ValueError: In case any of the requirement for the arguments are not respected.
        """

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

        self._initialize_weekend(weekend = weekend,
                                 min_weekend_traffic_variation = min_weekend_traffic_variation,
                                 max_weekend_traffic_variation = max_weekend_traffic_variation)

        self._initialize_black_friday(black_friday = black_friday,
                                      min_black_friday_traffic_variation = min_black_friday_traffic_variation,
                                      max_black_friday_traffic_variation = max_black_friday_traffic_variation)

        self._initialize_holiday_season(holiday_season = holiday_season,
                                        min_holiday_season_traffic_variation = min_holiday_season_traffic_variation,
                                        max_holiday_season_traffic_variation = max_holiday_season_traffic_variation)

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

    def _initialize_weekend(self,
                            weekend: bool,
                            min_weekend_traffic_variation: int,
                            max_weekend_traffic_variation: int):

        self._weekend = weekend

        if weekend:

            if max_weekend_traffic_variation < min_weekend_traffic_variation:
                raise ValueError('The maximum must be smaller than the minimum weekend traffic variation.')

            self._min_weekend_traffic_variation = min_weekend_traffic_variation
            self._max_weekend_traffic_variation = max_weekend_traffic_variation

    def _initialize_black_friday(self,
                                 black_friday: bool,
                                 min_black_friday_traffic_variation: int,
                                 max_black_friday_traffic_variation: int):

        self._black_friday = black_friday

        if black_friday:

            if max_black_friday_traffic_variation < min_black_friday_traffic_variation:
                raise ValueError('The maximum must be smaller than the minimum black friday traffic variation.')

            self._min_black_friday_traffic_variation = min_black_friday_traffic_variation
            self._max_black_friday_traffic_variation = max_black_friday_traffic_variation

    def _initialize_holiday_season(self,
                                   holiday_season: bool,
                                   min_holiday_season_traffic_variation: int,
                                   max_holiday_season_traffic_variation: int):

        self._holiday_season = holiday_season

        if holiday_season:

            if max_holiday_season_traffic_variation < min_holiday_season_traffic_variation:
                raise ValueError('The maximum must be smaller than the minimum holiday_season traffic variation.')

            self._min_holiday_season_traffic_variation = min_holiday_season_traffic_variation
            self._max_holiday_season_traffic_variation = max_holiday_season_traffic_variation

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

    def _jitter_weekend_traffic(self):

        day_num = len(self.data)
        day_size = self.data[0].size

        saturday = 5
        sunday = 6

        cur_date = self.start_date
        delta = timedelta(days = 1)

        for day_ind in range(day_num):

            if cur_date.weekday() in (saturday, sunday):
                shift = np.random.randint(low = self.min_weekend_traffic_variation * self.polling_interval,
                                          high = self.max_weekend_traffic_variation * self.polling_interval,
                                          size = day_size)
                self.data[day_ind] += shift

            cur_date += delta

    def _jitter_black_friday_traffic(self):

        day_num = len(self.data)
        day_size = self.data[0].size

        black_friday = datetime(year = 1, month = 11, day = 29)

        cur_date = self.start_date
        delta = timedelta(days = 1)

        for day_ind in range(day_num):

            if cur_date.month == black_friday.month and cur_date.day == black_friday.day:
                shift = np.random.randint(low = self.min_black_friday_traffic_variation * self.polling_interval,
                                          high = self.max_black_friday_traffic_variation * self.polling_interval,
                                          size = day_size)
                self.data[day_ind] += shift

            cur_date += delta

    def _jitter_holiday_season_traffic(self):

        day_num = len(self.data)
        day_size = self.data[0].size

        holiday_season = datetime(year = 1, month = 12, day = 1)

        holidays = (datetime(year = 1, month = 12, day = 24),
                    datetime(year = 1, month = 12, day = 25),
                    datetime(year = 1, month = 12, day = 31),
                    datetime(year = 1, month = 1, day = 1),
                    datetime(year = 1, month = 1, day = 2))

        cur_date = self.start_date
        delta = timedelta(days = 1)

        for day_ind in range(day_num):

            is_legal_holiday = any(cur_date.month == holiday.month and cur_date.day == holiday.day
                                   for holiday
                                   in holidays)

            is_holiday_season = not is_legal_holiday and cur_date.month == holiday_season.month

            if is_holiday_season or is_legal_holiday:
                shift = np.random.randint(low = int(self.min_holiday_season_traffic_variation *
                                                    self.polling_interval *
                                                    (0.5 if is_legal_holiday else 1.0)),
                                          high = int(self.max_holiday_season_traffic_variation *
                                                     self.polling_interval *
                                                     (1.5 if is_legal_holiday else 1.0)),
                                          size = day_size)
                self.data[day_ind] += shift

            cur_date += delta

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

        if self.weekend:
            self._jitter_weekend_traffic()

        if self.black_friday:
            self._jitter_black_friday_traffic()

        if self.holiday_season:
            self._jitter_holiday_season_traffic()

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
    def weekend(self) -> bool:
        return self._weekend

    @property
    def min_weekend_traffic_variation(self) -> int:
        return self._min_weekend_traffic_variation

    @property
    def max_weekend_traffic_variation(self) -> int:
        return self._max_weekend_traffic_variation

    @property
    def black_friday(self) -> bool:
        return self._black_friday

    @property
    def min_black_friday_traffic_variation(self) -> int:
        return self._min_black_friday_traffic_variation

    @property
    def max_black_friday_traffic_variation(self) -> int:
        return self._max_black_friday_traffic_variation

    @property
    def holiday_season(self) -> bool:
        return self._holiday_season

    @property
    def min_holiday_season_traffic_variation(self) -> int:
        return self._min_holiday_season_traffic_variation

    @property
    def max_holiday_season_traffic_variation(self) -> int:
        return self._max_holiday_season_traffic_variation

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days

    @property
    def data(self) -> List[np.ndarray]:
        return self._data

    def as_days(self) -> np.ndarray:
        data = []
        for day in self.data:
            hr_min_loads = self._expand_day(day)
            data.extend(hr_min_loads)
        data = np.array(data).astype(int)
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
        data = np.array(data).astype(int)
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
        data = np.array(data).astype(int)
        return data

    def as_years(self) -> np.ndarray:
        data = []
        curr_date = self.start_date
        delta = timedelta(days = 1)
        for day_ind, day in enumerate(self.data):
            mth = curr_date.month
            mthday = curr_date.day
            hr_min_loads = self._expand_day(day)
            mth_mnday_hr_min_loads = [(mth, mthday, hr, minute, load)
                                      for hr, minute, load
                                      in hr_min_loads]
            data.extend(mth_mnday_hr_min_loads)
            curr_date += delta
        data = np.array(data).astype(int)
        return data

    def as_lifetime(self) -> np.ndarray:
        data = []
        curr_date = self.start_date
        delta = timedelta(days = 1)
        for day_ind, day in enumerate(self.data):
            yr = curr_date.year
            mth = curr_date.month
            mthday = curr_date.day
            hr_min_loads = self._expand_day(day)
            yr_mnday_hr_min_loads = [(yr, mth, mthday, hr, minute, load)
                                     for hr, minute, load
                                     in hr_min_loads]
            data.extend(yr_mnday_hr_min_loads)
            curr_date += delta
        data = np.array(data).astype(int)
        return data

    def get_day(self, year: int, month: int, day: int) -> np.ndarray:

        date = datetime(year = year, month = month, day = day)

        if not self.start_date <= date or not date <= self.end_date:
            raise ValueError('The date must be between he start date and end date.')

        day_ind = (date - self.start_date).days
        return self.data[day_ind]

    def get_month(self, year: int, month: int) -> List[np.ndarray]:
        date = datetime(year = year, month = month, day = 1)

        too_early = (date.year < self.start_date.year or
                     (date.year == self.start_date.year and
                      date.month < self.start_date.month))
        too_late = (date.year > self.end_date.year or
                    (date.year == self.end_date.year and
                     date.month > self.end_date.month))

        if too_early or too_late:
            raise ValueError('The date must be between he start date and end date.')

        single_month = self.start_date.year == self.end_date.year and self.start_date.month == self.end_date.month
        first_month = not single_month and date.year == self.start_date.year and date.month == self.start_date.month
        last_month = not single_month and date.year == self.end_date.year and date.month == self.end_date.month

        if single_month:
            return self.data

        elif first_month:
            end_of_first_month = datetime(year = self.start_date.year, month = self.start_date.month + 1, day = 1)
            days_left_in_first_month = (end_of_first_month - self.start_date).days
            return self.data[:days_left_in_first_month]

        elif last_month:
            start_of_last_month = datetime(year = self.end_date.year, month = self.end_date.month, day = 1)
            days_left_in_last_month = (self.end_date - start_of_last_month).days
            return self.data[-days_left_in_last_month:]

        else:
            month_start_weekday, month_length = calendar.monthrange(year, month)
            day_ind = (date - self.start_date).days
            return self.data[day_ind:day_ind + month_length]

    def get_year(self, year: int) -> List[np.ndarray]:

        date = datetime(year = year, month = 1, day = 1)

        too_early = date.year < self.start_date.year
        too_late = date.year > self.end_date.year

        if too_early or too_late:
            raise ValueError('The date must be between he start date and end date.')

        single_year = self.start_date.year == self.end_date.year
        first_year = not single_year and date.year == self.start_date.year
        last_year = not single_year and date.year == self.end_date.year

        if single_year:
            return self.data

        elif first_year:
            end_of_first_year = datetime(year = self.start_date.year + 1, month = 1, day = 1)
            days_left_in_first_year = (end_of_first_year - self.start_date).days
            return self.data[:days_left_in_first_year]

        elif last_year:
            start_of_last_year = datetime(year = self.end_date.year, month = 1, day = 1)
            days_left_in_last_year = (self.end_date - start_of_last_year).days
            return self.data[-days_left_in_last_year:]

        else:
            year_length = (datetime(year = year + 1, month = 1, day = 1) - date).days
            day_ind = (date - self.start_date).days
            return self.data[day_ind:day_ind + year_length]

    def export_to_pdf(self,
                      filename: str,
                      weekday_names: List[str] = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'),
                      month_names: List[str] = ('Jan', 'Feb',
                                                'Mar', 'Apr', 'May',
                                                'Jun', 'Jul', 'Aug',
                                                'Sep', 'Oct', 'Nov',
                                                'Dec'),
                      verbose: bool = False):

        pdf = PdfPages(filename)

        cycle_mins = 24 * 60

        time_steps = list(zip(*[(mins // self.polling_interval, f'{mins // 60:02d}:{mins % 60:02d}')
                                for mins
                                in range(0, cycle_mins, cycle_mins // 24)]))

        time_steps = list(map(list, time_steps))

        time_steps[0] += [time_steps[0][-1] + time_steps[0][1]]
        time_steps[1] += ['24:00']

        time_stamps = time_steps[1][::12]
        time_steps = time_steps[0][::12]

        year = self.start_date.year
        month = self.start_date.month

        weeks_in_month, days_in_week = 6, 7

        while True:

            if verbose:
                print('\n ', year, month)

            fig = plt.figure(figsize = (20, 10))
            fig.subplots_adjust(wspace = 0.35, hspace = 0.35)

            fig.suptitle(f'{month_names[month - 1]}, {year}', fontsize = 20, fontweight = 'bold')

            month_start_weekday, month_length = calendar.monthrange(year, month)
            month_day_traffic = self.get_month(year, month)

            start_month = year == self.start_date.year and month == self.start_date.month

            if start_month:
                traffic_start_day = self.start_date.day - 1
            else:
                traffic_start_day = month_start_weekday

            for i, weekday_name in enumerate(weekday_names):
                plt.subplot(weeks_in_month, days_in_week, i + 1)
                plt.title(weekday_name, fontsize = 15, fontweight = 'bold')

            for i in range(traffic_start_day):
                plt.subplot(weeks_in_month, days_in_week, i + 1)
                plt.yticks([], [])
                plt.xticks([], [])

                if verbose:
                    print('_ ', end = '')
                    if (i + 1) % 7 == 0:
                        print()

            for day_ind, day in enumerate(month_day_traffic):
                plt.subplot(weeks_in_month, days_in_week, traffic_start_day + day_ind + 1)
                plt.plot(day, color = 'b', linewidth = 0.5)
                plt.ylim([0, max(self.max_traffic_at_peaks) * self.polling_interval * 1.25])
                plt.xticks(time_steps, time_stamps, rotation = 0)
                plt.grid(True, linestyle = '--')
                day_suff = ('st'
                            if day_ind % 10 == 0
                            else ('nd'
                                  if day_ind % 10 == 1
                                  else ('rd'
                                        if day_ind % 10 == 2
                                        else 'th')))
                plt.text(time_steps[-1] * 0.8,
                         max(self.max_traffic_at_peaks) * self.polling_interval * 1.1,
                         f'{day_ind + 1 + (traffic_start_day if start_month else 0)}{day_suff}',
                         fontweight = 'bold')

                if verbose:
                    print('X ', end = '')
                    if (traffic_start_day + day_ind + 1) % 7 == 0:
                        print()

            for i in range(len(month_day_traffic) + traffic_start_day, weeks_in_month * days_in_week):
                plt.subplot(weeks_in_month, days_in_week, i + 1)
                plt.yticks([], [])
                plt.xticks([], [])

                if verbose:
                    print('_ ', end = '')
                    if (i + 1) % 7 == 0:
                        print()

            pdf.savefig(fig)

            if year == self.end_date.year and month == self.end_date.month:
                break

            month += 1
            if month > 12:
                month = 1
                year += 1

        pdf.close()


if __name__ == '__main__':

    dataset = Dataset(start_date = (2014, 9, 15),
                      end_date = (2017, 9, 25),

                      polling_interval = 5,

                      min_traffic = 5,
                      max_traffic = 15,

                      peak_times = [(8, 30), (19, 30)],
                      peak_time_max_variation = 30,

                      peak_durations = [60, 120],
                      peak_duration_max_variation = 60,

                      max_traffic_at_peaks = [75, 150],
                      max_traffic_at_peak_variation = 50,

                      weekend = True,
                      min_weekend_traffic_variation = -10,
                      max_weekend_traffic_variation = 50,

                      black_friday = True,
                      min_black_friday_traffic_variation = 50,
                      max_black_friday_traffic_variation = 100,

                      holiday_season = True,
                      min_holiday_season_traffic_variation = 25,
                      max_holiday_season_traffic_variation = 50)

    dataset.export_to_pdf('out.pdf',
                          verbose = True)









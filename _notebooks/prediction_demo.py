import datetime
from time import clock
import numpy as np
from prediction.dataset import Dataset
from prediction import XgbTimeSeriesRegressor

import math
import operator as op
import random
import sys

from deap import gp
from deap import base
from deap import tools
from deap import creator
from deap import algorithms

import pathos.multiprocessing as mp

from typing import *

from prediction._ga_utils import FitnessMax, Individual


if __name__ == '__main__':

    dataset_params = {'polling_interval': 5,

                      'min_traffic': 5,
                      'max_traffic': 15,

                      'peak_times': [(8, 30), (19, 30)],
                      'peak_time_max_variation': 30,

                      'peak_durations': [60, 120],
                      'peak_duration_max_variation': 30,

                      'max_traffic_at_peaks': [100, 150],
                      'max_traffic_at_peak_variation': 50,

                      'weekend': True,
                      'min_weekend_traffic_variation': -15,
                      'max_weekend_traffic_variation': 50,

                      'black_friday': True,
                      'min_black_friday_traffic_variation': 50,
                      'max_black_friday_traffic_variation': 150,

                      'holiday_season': True,
                      'min_holiday_season_traffic_variation': 30,
                      'max_holiday_season_traffic_variation': 100}

    print(f'Starting...')
    start = clock()

    dataset = Dataset(start_date = (2004, 9, 15),
                      end_date = (2015, 9, 25),
                      **dataset_params)

    dataset.generate_data()

    dataset2 = Dataset.from_data(data = dataset.data,
                                 start_date = (dataset.start_date.year,
                                               dataset.start_date.month,
                                               dataset.start_date.day),
                                 end_date = (dataset.end_date.year,
                                             dataset.end_date.month,
                                             dataset.end_date.day),
                                 polling_interval = dataset.polling_interval)

    dataset = dataset2

    test_dataset = Dataset(start_date = (2015, 9, 25),
                           end_date = (2018, 12, 31),
                           **dataset_params)

    test_dataset.generate_data()

    end = clock()
    print(f'Created test datasets in {end - start:.4f} s...')
    start = clock()

    ((days_x, days_y),
     (weeks_x, weeks_y),
     (months_x, months_y),
     (years_x, years_y),
     (lifetime_x, lifetime_y)) = XgbTimeSeriesRegressor._partition_dataset(dataset2)

    ((val_days_x, val_days_y),
     (val_weeks_x, val_weeks_y),
     (val_months_x, val_months_y),
     (val_years_x, val_years_y),
     (val_lifetime_x, val_lifetime_y)) = XgbTimeSeriesRegressor._partition_dataset(test_dataset)

    end = clock()
    print(f'Partitioned datasets in {end - start:.4f} s...')
    start = clock()

    model = XgbTimeSeriesRegressor()
    # model = XgbTimeSeriesRegressor.load_model(r'.\model', 'alpha')

    end = clock()
    print(f'Created model in {end - start:.4f} s...')
    start = clock()

    model.fit(lifetime_x, lifetime_y,
              val_lifetime_x, val_lifetime_y,
              fit_combiner_func = True,
              njobs = 4,
              verbose = 3)

    end = clock()
    print(f'Trained model in {end - start:.4f} s...')
    start = clock()

    res = model.score_on_dataset(dataset)
    print(f'Score for average of predictions on training is {res:.4f}')
    res = model.score_on_dataset(test_dataset)
    print(f'Score for average of predictions on validation is {res:.4f}')

    end = clock()
    print(f'Evaluated model in {end - start:.4f} s...')

    now = datetime.datetime.now()
    pref = f'{now.year:04}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}'

    model.save_model(folder = r'.\model', prefix = pref)











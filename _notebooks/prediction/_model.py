import os
import dill
import random
import operator as op

import numpy as np
import xgboost as xgb

from time import clock
from datetime import datetime

from prediction.dataset import Dataset
from prediction._ga_utils import genetic_algorithm
from prediction._ga_utils import FitnessMax
from prediction._ga_utils import Individual

from sklearn.metrics import r2_score

from deap import gp
from deap import base
from deap import tools
from deap import creator
from deap import algorithms

import pathos.multiprocessing as mp


from typing import *


class XgbTimeSeriesRegressor:

    _day_model: xgb.XGBRegressor
    _week_model: xgb.XGBRegressor
    _month_model: xgb.XGBRegressor
    _year_model: xgb.XGBRegressor
    _lifetime_model: xgb.XGBRegressor

    _combiner_func: Callable

    def __init__(self, combiner_func: Callable = None):

        self._day_model = xgb.XGBRegressor(max_depth = 3,
                                           n_estimators = 250)

        self._week_model = xgb.XGBRegressor(max_depth = 7,
                                            n_estimators = 250)

        self._month_model = xgb.XGBRegressor(max_depth = 7,
                                             n_estimators = 250)

        self._year_model = xgb.XGBRegressor(max_depth = 7,
                                            n_estimators = 250)

        self._lifetime_model = xgb.XGBRegressor(max_depth = 5,
                                                n_estimators = 250)

        if combiner_func is None:

            def combine(yr: int,
                        mon: int,
                        mond: int,
                        wkd: int,
                        hr: int,
                        mi: int,
                        d_p: float,
                        w_p: float,
                        m_p: float,
                        y_p: float,
                        l_p: float) -> float:

                return sum([weight * pred
                            for weight, pred
                            in zip([0.2] * 5,
                                   (d_p, w_p, m_p, y_p, l_p))])

            self._combiner_func = combine

    @property
    def day_model(self) -> xgb.XGBRegressor:
        return self._day_model

    @property
    def week_model(self) -> xgb.XGBRegressor:
        return self._week_model

    @property
    def month_model(self) -> xgb.XGBRegressor:
        return self._month_model

    @property
    def year_model(self) -> xgb.XGBRegressor:
        return self._year_model

    @property
    def lifetime_model(self) -> xgb.XGBRegressor:
        return self._lifetime_model

    @staticmethod
    def _to_dataset(x: np.ndarray, y: np.ndarray) -> Dataset:

        dates = [datetime(year = yr,
                          month = mo,
                          day = da,
                          hour = hr,
                          minute = mi)
                 for yr, mo, da, hr, mi
                 in x]

        start_date = min(dates)
        end_date = max(dates)

        tail_dates = dates.copy()
        tail_dates.remove(start_date)
        next_date = min(tail_dates)
        polling_interval = int((next_date - start_date).total_seconds()) // 60

        # y = y[:, np.newaxis]
        # x_y = np.concatenate((x, y), axis = 1)

        y = y.reshape(-1, 1440 // polling_interval)

        data = Dataset.from_data(data = y,
                                 start_date = start_date,
                                 end_date = end_date,
                                 polling_interval = polling_interval)

        return data

    @staticmethod
    def _partition_dataset(data: Dataset) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                   Tuple[np.ndarray, np.ndarray],
                                                   Tuple[np.ndarray, np.ndarray],
                                                   Tuple[np.ndarray, np.ndarray],
                                                   Tuple[np.ndarray, np.ndarray]]:

        days = data.as_days()
        weeks = data.as_weeks()
        months = data.as_months()
        years = data.as_years()
        lifetime = data.as_lifetime()

        days_x, days_y = days[:, :-1], days[:, -1]
        weeks_x, weeks_y = weeks[:, :-1], weeks[:, -1]
        months_x, months_y = months[:, :-1], months[:, -1]
        years_x, years_y = years[:, :-1], years[:, -1]
        lifetime_x, lifetime_y = lifetime[:, :-1], lifetime[:, -1]

        return ((days_x, days_y),
                (weeks_x, weeks_y),
                (months_x, months_y),
                (years_x, years_y),
                (lifetime_x, lifetime_y))

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            val_x: np.ndarray = None,
            val_y: np.ndarray = None,
            fit_combiner_func: bool = False,
            njobs: int = 1,
            verbose: int = 0):

        if verbose:
            print('Creating dataset structure...')
            start = clock()

        data = self._to_dataset(x, y)

        has_val_data = val_x is not None and val_y is not None

        if has_val_data:

            if verbose:
                stop = clock()
                print(f'Done in {stop - start: .4f} s.')
                print('Creating validation dataset structure...')
                start = clock()

            val_data = self._to_dataset(val_x, val_y)

        else:
            val_data = None

        if verbose:
            stop = clock()
            print(f'Done in {stop - start: .4f} s.')

        self.fit_dataset(data, val_data, verbose)

        if fit_combiner_func:
            self.fit_combiner_func(x, y,
                                   val_x, val_y,
                                   50, 50, njobs,
                                   verbose = verbose)

    def fit_dataset(self,
                    data: Dataset,
                    val_data: Dataset = None,
                    verbose: int = 0):

        has_val_data = val_data is not None

        if verbose:
            print('Partitioning data...')
            start = clock()

        ((days_x, days_y),
         (weeks_x, weeks_y),
         (months_x, months_y),
         (years_x, years_y),
         (lifetime_x, lifetime_y)) = self._partition_dataset(data)

        if has_val_data:

            if verbose:
                stop = clock()
                print(f'Done in {stop - start: .4f} s.')
                print('Partitioning validation data...')
                start = clock()

            ((val_days_x, val_days_y),
             (val_weeks_x, val_weeks_y),
             (val_months_x, val_months_y),
             (val_years_x, val_years_y),
             (val_lifetime_x, val_lifetime_y)) = self._partition_dataset(val_data)

        if verbose:
            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print('Training day recurrence model...')
            start = clock()

        if has_val_data:
            self.day_model.fit(days_x, days_y,
                               eval_set = ((val_days_x, val_days_y),),
                               eval_metric = 'rmse',
                               early_stopping_rounds = 3,
                               verbose = verbose == 2)
        else:
            self.day_model.fit(days_x, days_y)

        if verbose:
            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print('Training week recurrence model...')
            start = clock()

        if has_val_data:
            self.week_model.fit(weeks_x, weeks_y,
                                eval_set = ((val_weeks_x, val_weeks_y),),
                                eval_metric = 'rmse',
                                early_stopping_rounds = 3,
                                verbose = verbose == 2)
        else:
            self.week_model.fit(weeks_x, weeks_y)

        if verbose:
            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print('Training month recurrence model...')
            start = clock()

        if has_val_data:
            self.month_model.fit(months_x, months_y,
                                 eval_set = ((val_months_x, val_months_y),),
                                 eval_metric = 'rmse',
                                 early_stopping_rounds = 3,
                                 verbose = verbose == 2)
        else:
            self.month_model.fit(months_x, months_y)

        if verbose:
            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print('Training year recurrence model...')
            start = clock()

        if has_val_data:
            self.year_model.fit(years_x, years_y,
                                eval_set = ((val_years_x, val_years_y),),
                                eval_metric = 'rmse',
                                early_stopping_rounds = 3,
                                verbose = verbose == 2)
        else:
            self.year_model.fit(years_x, years_y)

        if verbose:
            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print('Training lifetime model...')
            start = clock()

        if has_val_data:
            self.lifetime_model.fit(lifetime_x, lifetime_y,
                                    eval_set = ((val_lifetime_x, val_lifetime_y),),
                                    eval_metric = 'rmse',
                                    early_stopping_rounds = 3,
                                    verbose = verbose == 2)
        else:
            self.lifetime_model.fit(lifetime_x, lifetime_y)

        if verbose:

            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print('Computing R^2 scores on training data...')
            start = clock()

            day_r2 = self.day_model.score(days_x, days_y)
            week_r2 = self.week_model.score(weeks_x, weeks_y)
            month_r2 = self.month_model.score(months_x, months_y)
            year_r2 = self.year_model.score(years_x, years_y)
            lifetime_r2 = self.lifetime_model.score(lifetime_x, lifetime_y)
            combined_r2 = self.score(lifetime_x, lifetime_y)

            stop = clock()
            print(f'Done in {stop - start: .4f} s.')
            print(f'R^2 scores for each model based on training data:\n'
                  f'Day     : {day_r2:6.4f}\n'
                  f'Week    : {week_r2:6.4f}\n'
                  f'Month   : {month_r2:6.4f}\n'
                  f'Year    : {year_r2:6.4f}\n'
                  f'Life    : {lifetime_r2:6.4f}\n'
                  f'Combined: {combined_r2:6.4f}')

            if has_val_data:

                print('Computing R^2 scores on validation data...')
                start = clock()

                val_day_r2 = self.day_model.score(val_days_x, val_days_y)
                val_week_r2 = self.week_model.score(val_weeks_x, val_weeks_y)
                val_month_r2 = self.month_model.score(val_months_x, val_months_y)
                val_year_r2 = self.year_model.score(val_years_x, val_years_y)
                val_lifetime_r2 = self.lifetime_model.score(val_lifetime_x, val_lifetime_y)
                val_combined_r2 = self.score(val_lifetime_x, val_lifetime_y)

                stop = clock()
                print(f'Done in {stop - start: .4f} s.')
                print(f'R^2 scores for each model based on validation data:\n'
                      f'Day     : {val_day_r2:6.4f}\n'
                      f'Week    : {val_week_r2:6.4f}\n'
                      f'Month   : {val_month_r2:6.4f}\n'
                      f'Year    : {val_year_r2:6.4f}\n'
                      f'Life    : {val_lifetime_r2:6.4f}\n'
                      f'Combined: {val_combined_r2:6.4f}')

    def fit_combiner_func(self,
                          x: np.ndarray,
                          y: np.ndarray,
                          val_x: np.ndarray,
                          val_y: np.ndarray,
                          gens: int,
                          popsize: int,
                          njobs: int = 1,
                          verbose: int = 0):

        def op_div(left, right):
            try:
                return left / right
            except ZeroDivisionError:
                return 0

        pset = gp.PrimitiveSet('MAIN', 11)

        pset.addPrimitive(op.add, 2)
        pset.addPrimitive(op.sub, 2)

        pset.addPrimitive(op.mul, 2)
        pset.addPrimitive(op_div, 2, name = 'div')

        pset.addPrimitive(op.neg, 1)

        for i in range(-3, 3 + 1):
            pset.addTerminal(i)

        pset.renameArguments(ARG0  = 'year',
                             ARG1  = 'mon',
                             ARG2  = 'mond',
                             ARG3  = 'wkd',
                             ARG4  = 'hr',
                             ARG5  = 'mi',
                             ARG6  = 'day_pred',
                             ARG7  = 'wk_pred',
                             ARG8  = 'mon_pred',
                             ARG9  = 'yr_pred',
                             ARG10 = 'life_pred')

        # Create the toolbox.

        toolbox = base.Toolbox()

        toolbox.register('expr', gp.genHalfAndHalf, pset = pset, min_ = 1, max_ = 3)
        toolbox.register('individual', tools.initIterate, Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('compile', gp.compile, pset = pset)

        def eval_combiner(individual,
                          compile_func: Callable,
                          model: XgbTimeSeriesRegressor,
                          x: np.ndarray,
                          y: np.ndarray,
                          verbose: int = 0):

            combiner_func = compile_func(expr = individual)
            model._combiner_func = combiner_func
            score = model.score(x, y)
            score = -score if score > 1 else score

            if verbose > 2 or (verbose == 2 and score > 0):
                print(f'R^2_a score {score:>8.4f} for expr {individual}')

            return score,

        toolbox.register('evaluate', eval_combiner,
                         model = self,
                         compile_func = toolbox.compile,
                         x = x, y = y,
                         verbose = verbose)

        toolbox.register('select', tools.selTournament, tournsize = 3)

        toolbox.register('mate', gp.cxOnePoint)
        toolbox.decorate('mate', gp.staticLimit(key = op.attrgetter('height'), max_value = 20))

        toolbox.register('small_expr', gp.genHalfAndHalf, min_ = 1, max_ = 3)
        toolbox.register('mutate', gp.mutUniform, expr = toolbox.small_expr, pset = pset)
        toolbox.decorate('mutate', gp.staticLimit(key = op.attrgetter('height'), max_value = 20))

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fit.register('Average R^2'.ljust(15), np.mean)
        stats_fit.register('Maximum R^2'.ljust(15), np.max)

        def validation_r2(population: List[gp.PrimitiveTree],
                          model: XgbTimeSeriesRegressor,
                          x: np.ndarray,
                          y: np.ndarray) -> float:

            best = max(population, key = lambda ind: ind.fitness)
            model._combiner_func = toolbox.compile(expr = best)
            score = model.score(x, y)

            return score

        stats_val = tools.Statistics()
        stats_val.register('Validation R^2'.ljust(15), validation_r2, model = self, x = val_x, y = val_y)

        stats_size = tools.Statistics(len)
        stats_size.register('Average'.ljust(15), np.mean)
        stats_size.register('Maximum'.ljust(15), np.max)

        mstats = tools.MultiStatistics(fitness = stats_fit, val = stats_val, size = stats_size)

        hof = tools.HallOfFame(1)

        toolbox.mutations = [toolbox.mutate]
        toolbox.pop_init = toolbox.population

        def disp(pop: list, log: tools.Logbook, hof: tools.HallOfFame):
            print()
            print(log.stream)
            print()

        pool = mp.Pool(njobs)

        sol, log = genetic_algorithm(toolbox = toolbox,
                                     cxpb = 0.5,
                                     mutpb = 0.25,
                                     popsize = popsize,
                                     popreplace = 0.0,
                                     tournsize = max(3, popsize // 100),
                                     gens = gens,
                                     elitism = max(3, popsize // 100),
                                     njobs = njobs,
                                     pool = pool,
                                     creator = creator,
                                     hof = hof,
                                     stats = mstats,
                                     verbose = verbose > 0,
                                     dispfunc = disp)

        self._combiner_func = toolbox.compile(hof.items[0])

        if verbose:
            print('Best individual:')
            print(hof.items[0])
            print('R^2 score on training data:', self.score(x, y))
            print('R^2 score on validation data:', self.score(val_x, val_y))

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return r2_score(y, y_pred)

    def score_on_dataset(self, data: Dataset) -> float:
        (_, _), (_, _), (_, _), (_, _), (x, y) = self._partition_dataset(data)
        return self.score(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:

        dates = [datetime(year = yr,
                          month = mo,
                          day = da,
                          hour = hr,
                          minute = mi)
                 for yr, mo, da, hr, mi
                 in x]

        days_x = np.array([(date.hour, date.minute)
                           for date
                           in dates])

        weeks_x = np.array([(date.weekday(), date.hour, date.minute)
                            for date
                            in dates])

        months_x = np.array([(date.day, date.hour, date.minute)
                             for date
                             in dates])

        years_x = np.array([(date.month, date.day, date.hour, date.minute)
                            for date
                            in dates])

        lifetime_x = np.array([(date.year, date.month, date.day, date.hour, date.minute)
                               for date
                               in dates])

        day_pred = self.day_model.predict(days_x)
        week_pred = self.week_model.predict(weeks_x)
        month_pred = self.month_model.predict(months_x)
        year_pred = self.year_model.predict(years_x)
        lifetime_pred = self.lifetime_model.predict(lifetime_x)

        weekdays = [d.weekday() for d in dates]

        pred = np.array([self._combiner_func(np.asscalar(yr),
                                             np.asscalar(mo),
                                             np.asscalar(mda),
                                             wda,
                                             np.asscalar(hr),
                                             np.asscalar(mi),
                                             np.asscalar(dap),
                                             np.asscalar(wep),
                                             np.asscalar(mop),
                                             np.asscalar(yrp),
                                             np.asscalar(lip))
                         for yr, mo, mda, wda, hr, mi, dap, wep, mop, yrp, lip
                         in zip(x[:, 0], x[:, 1], x[:, 2], weekdays, x[:, 3], x[:, 4],
                                day_pred, week_pred, month_pred, year_pred, lifetime_pred)])

        return pred

    def save_model(self, folder: str, prefix: str):

        day_model_path = os.path.join(folder, prefix + '_day_model.h5')
        week_model_path = os.path.join(folder, prefix + '_week_model.h5')
        month_model_path = os.path.join(folder, prefix + '_month_model.h5')
        year_model_path = os.path.join(folder, prefix + '_year_model.h5')
        lifetime_model_path = os.path.join(folder, prefix + '_lifetime_model.h5')
        combiner_func_path = os.path.join(folder, prefix + '_combiner_func.pkl')

        self.day_model.save_model(day_model_path)
        self.week_model.save_model(week_model_path)
        self.month_model.save_model(month_model_path)
        self.year_model.save_model(year_model_path)
        self.lifetime_model.save_model(lifetime_model_path)

        with open(combiner_func_path, 'wb') as combiner_func_file:
            dill.dump(self._combiner_func, combiner_func_file)

    @staticmethod
    def load_model(folder: str, prefix: str) -> 'XgbTimeSeriesRegressor':

        day_model_path = os.path.join(folder, prefix + '_day_model.h5')
        week_model_path = os.path.join(folder, prefix + '_week_model.h5')
        month_model_path = os.path.join(folder, prefix + '_month_model.h5')
        year_model_path = os.path.join(folder, prefix + '_year_model.h5')
        lifetime_model_path = os.path.join(folder, prefix + '_lifetime_model.h5')
        combiner_func_path = os.path.join(folder, prefix + '_combiner_func.pkl')

        model = XgbTimeSeriesRegressor()

        model.day_model.load_model(day_model_path)
        model.week_model.load_model(week_model_path)
        model.month_model.load_model(month_model_path)
        model.year_model.load_model(year_model_path)
        model.lifetime_model.load_model(lifetime_model_path)

        with open(combiner_func_path, 'rb') as combiner_func_file:
            model._combiner_func = dill.load(combiner_func_file)

        return model

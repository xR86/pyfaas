import os
import dill
import datetime

from flask import Flask
from flask import request

import numpy as np
from prediction import XgbTimeSeriesRegressor
from prediction.dataset import Dataset


app = Flask(__name__)


@app.route('/predict', methods = ['GET'])
def predict():

    # Check param existence.

    if 'year' not in request.args:
        return 'ERR_NO_YEAR_SPECIFIED'
    if 'month' not in request.args:
        return 'ERR_NO_MONTH_SPECIFIED'
    if 'day' not in request.args:
        return 'ERR_NO_DAY_SPECIFIED'
    if 'hour' not in request.args:
        return 'ERR_NO_HOUR_SPECIFIED'
    if 'minute' not in request.args:
        return 'ERR_NO_MINUTE_SPECIFIED'

    # Get param values.

    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    minute = request.args.get('minute')

    # Cast params to int.

    try:
        year = int(year)
    except ValueError:
        return 'ERR_YEAR_NOT_INT'

    try:
        month = int(month)
    except ValueError:
        return 'ERR_MONTH_NOT_INT'

    try:
        day = int(day)
    except ValueError:
        return 'ERR_DAY_NOT_INT'

    try:
        hour = int(hour)
    except ValueError:
        return 'ERR_HOUR_NOT_INT'

    try:
        minute = int(minute)
    except ValueError:
        return 'ERR_MINUTE_NOT_INT'

    # Compose data to predict.

    data = np.array([[year, month, day, hour, minute]])

    # Get most recent model.

    most_recent_model_files = sorted(os.listdir(r'.\model'))[-6:]
    most_recent_model_date = most_recent_model_files[0][:16]

    if not all(most_recent_model_date == model_file[:16]
               for model_file
               in most_recent_model_files):

        return 'ERR_NEW_MODEL_BEING_WRITTEN'

    else:

        pretty_model_date = most_recent_model_date.replace('_', '-').split('-')
        pretty_model_date = (f'{pretty_model_date[0]}.{pretty_model_date[1]}.{pretty_model_date[2]} '
                             f'{pretty_model_date[3]}:{pretty_model_date[4]}')

        pretty_data_date = (f'{data[0, 0]}.{data[0, 1]}.{data[0, 2]} '
                            f'{data[0, 3]}:{data[0, 4]}')

        print(f'Requested prediction for date {pretty_data_date}. Most recent model is {pretty_model_date}.')

    model = XgbTimeSeriesRegressor.load_model(r'.\model', most_recent_model_date)

    # Do prediction.

    pred = model.predict(data)

    return str(pred[0])


@app.route('/train', methods = ['POST'])
def train():

    # Get the data.

    if 'data' not in request.values:
        return 'ERR_NO_DATA'

    data: np.ndarray = dill.loads(request.values.get('data'))

    x = data[:, :-1]
    y = data[:, -1]

    # Get whether to optimize combiner func or not.

    optimize_combiner = request.values.get('optimize_combiner', False)

    # Get the current date.

    now = datetime.datetime.now()
    pref = f'{now.year:04}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}'

    # Fit.

    model = XgbTimeSeriesRegressor()

    model.fit(x, y,
              fit_combiner_func = optimize_combiner,
              njobs = 2,
              verbose = 3)

    model.save_model(r'.\model', pref)

    return 'OK'


if __name__ == "__main__":
    app.run()

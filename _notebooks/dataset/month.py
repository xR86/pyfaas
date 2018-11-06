import calendar
import itertools

from operator import add
from operator import floordiv

import numpy as np

import toolz
from toolz import flip
from toolz import curry
from toolz import compose


import matplotlib
matplotlib.use('WXAgg')

import matplotlib.pyplot as plt
plt.ion()

import dataset.day
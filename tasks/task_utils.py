#!/usr/bin/env python3
"""
# to automatically reload modules who's content has changed
%load_ext autoreload
%autoreload 2
# configure matplotlib
%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
"""


import time
import numpy as np
import GPy
import sklearn.gaussian_process as sk_gp
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='numpy.dtype size changed') # annoying seaborn bug
    import seaborn as sns; sns.set()

import ipywidgets as widgets
from IPython.display import display
from IPython.core.debugger import set_trace




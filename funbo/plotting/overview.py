#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# local imports
from .utils import multidimensional_integrate, integer_ticks


def plot_consecutive_distance(optimiser, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    trials = optimiser.trials
    ts = np.arange(1, len(trials))
    ds = [multidimensional_integrate(lambda x: np.abs(trials[t-1].f(x)-trials[t].f(x)), optimiser.domain_bounds) for t in ts]
    ax.plot(ts, ds, marker='o', label='$\int|f_{i-1}(x)-f_i(x)|dx$')
    ax.set_title('Adjacent Distance')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Area')
    ax.legend()
    integer_ticks(ax.xaxis)


def plot_convergence(optimiser, best_R_g, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    xs = [t.trial_num for t in optimiser.trials]
    ys = [t.R_g for t in optimiser.trials]
    ax.plot(xs, ys, zorder=2, label='$R_g$')
    _scatter_trials(optimiser.trials, xs, ys, ax)
    chooser = np.max if optimiser.is_maximising() else np.min
    best_ys = [chooser(ys[:x+1]) for x in xs]
    ax.plot(xs, best_ys, zorder=1, label='$R_g^*$')
    _scatter_trials(optimiser.trials, xs, best_ys, ax)
    if best_R_g is not None:
        ax.axhline(y=best_R_g, linestyle=':', color='grey', label='best $R_g$')
    ax.legend()
    ax.set_ylabel('$R_g$')
    ax.set_xlabel('Trial')
    ax.set_title('Convergence')
    integer_ticks(ax.xaxis)

def _scatter_trials(trials, xs, ys, ax):
    c = ['violet' if t.surrogate is None else '#4c72b0' for t in trials]
    ax.scatter(xs, ys, c=c, zorder=5)

def plot_timings(optimiser, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    xs = [t.trial_num for t in optimiser.trials]
    ts = [t.timing_info for t in optimiser.trials]
    ax.plot(xs, [t.total for t in ts], ':', label='total')
    _scatter_trials(optimiser.trials, xs, [t.total for t in ts], ax)
    ax.stackplot(xs,
                 [t.fitting or 0 for t in ts],
                 [t.extraction or 0 for t in ts],
                 [t.evaluation for t in ts],
                 labels=['fitting', 'extraction', 'evaluation'])
    ax.set_title('Timings')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Seconds')
    ax.legend()
    integer_ticks(ax.xaxis)


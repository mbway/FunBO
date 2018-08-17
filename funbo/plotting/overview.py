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


def plot_convergence(optimiser, optimal_R_g, ax=None):
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
    if optimal_R_g is not None:
        ax.axhline(y=optimal_R_g, linestyle=':', color='grey', label='optimal $R_g$')
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
    ts = optimiser.trials
    xs = [t.trial_num for t in ts]
    ax.plot(xs, [t.duration for t in ts], ':', label='trial total')
    _scatter_trials(ts, xs, [t.duration for t in ts], ax)
    ax.stackplot(xs,
                 [t.selection.fitting.duration if t.is_bayes() else 0 for t in ts],
                 [t.selection.extraction.acq_total_time if t.is_bayes() else 0 for t in ts],
                 [t.selection.extraction.get_overhead_duration() if t.is_bayes() else 0 for t in ts],
                 [t.evaluation.duration for t in ts],
                 labels=['fitting', 'extraction > query acquisition', 'extraction > overhead', 'evaluation'])
    ax.set_title('Timings')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Seconds')
    ax.legend()
    integer_ticks(ax.xaxis)

def plot_trial_quantities(optimiser, quantities, ax=None):
    """
    Args:
        quantities: a dictionary of {label: f} where f is a function which takes
            a trial and returns a quantity to plot

    eg:
        plot_trial_quantities(opt, {
            'duration': lambda t: t.duration,
            'data set': lambda t: t.selection.fitting.data_set_size if t.is_bayes() else 0
        })
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    ts = optimiser.trials
    xs = [t.trial_num for t in ts]
    for label, f in quantities.items():
        ax.plot(xs, [f(t) for t in ts], label=label)
    ax.set_title('Trial Quantities')
    ax.set_xlabel('Trial')
    ax.legend()
    integer_ticks(ax.xaxis)


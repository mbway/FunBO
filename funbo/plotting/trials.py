#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from .utils import unzip, get_reward_cmap

def plot_trials(optimiser, trials, true_best=None, color_by_reward=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    xmin, xmax = optimiser.domain_bounds[0]
    xs = np.linspace(xmin, xmax, num=200)
    if true_best is not None:
        ax.plot(xs, [true_best(x) for x in xs], 'k--', label='to fit')
        ax.legend()

    if color_by_reward:
        R_g_color = get_reward_cmap([t.R_g for t in trials], optimiser.is_minimising())
        R_l_color = get_reward_cmap(np.concatenate([np.array([r for x, r in t.R_ls]) for t in trials]), optimiser.is_minimising())

    for t in trials:
        ys = [t.f(x) for x in xs]
        ax.plot(xs, ys, color=R_g_color.to_rgba(t.R_g) if color_by_reward else 'grey', alpha=0.3)
        # sampled locations where local rewards were given
        s_xs, s_rs = unzip(t.R_ls)
        s_ys = [t.f(x) for x in s_xs]
        c = R_l_color.to_rgba(s_rs) if color_by_reward else (0.2, 0.2, 0.2)
        ax.scatter(s_xs, s_ys, c=c, s=5)



#TODO: plot how the best trial changes over time


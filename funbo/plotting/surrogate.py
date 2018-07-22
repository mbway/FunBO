
import numpy as np
import matplotlib.pyplot as plt

from .utils import *
from .trials import plot_trials

try:
    import plotly.offline as ply
    import plotly.graph_objs as go

    if in_jupyter():
        print('plotly setup')
        ply.init_notebook_mode(connected=False)
except ImportError:
    ply = None



def plot_trial_area(optimiser, trial, true_best, quiet=False, ax=None):
    """
    """
    xmin, xmax = optimiser.domain_bounds[0]
    if not quiet:
        print('area between: {}'.format(multidimensional_integrate(lambda x: abs(true_best(x)-trial.f(x)), optimiser.domain_bounds)))
        print('R_g = {}'.format(trial.R_g))
        print('R_ls = {}'.format(format_list([r for x, r in trial.R_ls], 3)))
        print('f = {}'.format(trial.f))
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.linspace(xmin, xmax, num=200)
    fs = [trial.f(x) for x in xs]

    ax.plot(xs, fs, color='salmon', label='f')
    ax.scatter(trial.f.control_xs, trial.f.control_ys, color='salmon', s=10, label='f control points')

    fits = [true_best(x) for x in xs]
    ax.plot(xs, fits, linestyle='--', color='k', label='to fit')
    ax.fill_between(xs, [min(a, b) for a,b in zip(fits, fs)], [max(a,b) for a,b in zip(fits, fs)], alpha=0.2)

    # sampled locations where local rewards were given
    if trial.R_ls:
        s_xs, s_rs = unzip(trial.R_ls)
        s_ys = [trial.f(x) for x in s_xs]
        ax.scatter(s_xs, s_ys, c='black', s=10, zorder=10, label='local rewards')

    ax.legend()


def plot_surrogate_with_trials(optimiser, trial_num, true_best=None, ylim=None, midpoint_fraction=None):
    if trial_num < 0:
        trial_num = len(optimiser.trials) + trial_num
    t = optimiser.trials[trial_num]
    fig = plot_surrogate(optimiser, t.surrogate, ylim, midpoint_fraction)
    plot_trials(optimiser, optimiser.trials[:trial_num+1], true_best, color_by_reward=False, ax=fig.axes[0])
    return fig

def plot_surrogate(optimiser, surrogate, ylim=None, midpoint_fraction=None):
    assert surrogate is not None, 'no surrogate'
    fig, ax = plt.subplots(figsize=(20, 10))
    # undo seaborn styling
    ax.grid(False)
    ax.set_facecolor((1, 1, 1))

    x_name = optimiser.domain_names[0]
    xmin, xmax = optimiser.domain_bounds[0]
    xs = np.linspace(xmin, xmax, num=100)

    if ylim is not None:
        ymin, ymax = ylim
        y_margin = 0
    else:
        ymin, ymax = optimiser.range_bounds
        y_margin = (ymax-ymin)*0.1

    ys = np.linspace(ymin-y_margin, ymax+y_margin, num=100)
    ax.margins(0, 0)

    grid = np.meshgrid(xs, ys)
    mus_points, var_points = surrogate.predict(grid_to_points(grid))
    mus_grid = points_to_grid(mus_points, grid)

    cmap = 'viridis' if optimiser.is_maximising() else 'viridis_r'
    if midpoint_fraction is None:
        norm = None
    else:
        norm = utils.MidpointNorm(np.min(mus_points), np.max(mus_points), midpoint=None, midpoint_fraction=midpoint_fraction)
    im = ax.pcolormesh(grid[0], grid[1], mus_grid, cmap=cmap, norm=norm)
    c = fig.colorbar(im, ax=ax)
    c.set_label(r'$R_l({},f({}))$'.format(x_name, x_name))

    ax.set_xlabel('${}$'.format(x_name))
    ax.set_ylabel(r'$f({})$'.format(x_name))

    return fig

def plot_surrogate_3D(optimiser, surrogate, show_var=True, flip_z=False):
    assert ply is not None, 'plotly not imported'
    assert surrogate is not None, 'no surrogate'

    x_name = optimiser.domain_names[0]
    xmin, xmax = optimiser.domain_bounds[0]
    xs = np.linspace(xmin, xmax, num=100)

    ymin, ymax = optimiser.range_bounds
    y_margin = (ymax-ymin)*0.1
    ys = np.linspace(ymin-y_margin, ymax+y_margin, num=100)

    grid = np.meshgrid(xs, ys)
    mus_points, var_points = surrogate.predict(grid_to_points(grid))
    mus_grid = points_to_grid(mus_points, grid)
    if flip_z:
        mus_grid = -mus_grid

    data = [
        go.Surface(x=grid[0], y=grid[1], z=mus_grid, colorscale='Viridis', reversescale=optimiser.is_minimising() != flip_z),
    ]
    if show_var:
        sig_grid = points_to_grid(np.sqrt(np.clip(var_points, 0, np.inf)), grid)
        n_sig = 2
        color_row = ['#0000ff'] * grid[0].shape[1]
        colors = [color_row] * grid[0].shape[0]
        data.extend([
            go.Surface(x=grid[0], y=grid[1], z=mus_grid + n_sig*sig_grid, surfacecolor=colors, opacity=0.3, showscale=False),
            #go.Surface(x=grid[0], y=grid[1], z=mus_grid - n_sig*sig_grid, surfacecolor=colors, opacity=0.3, showscale=False),
        ])

    layout = go.Layout(
        title='3D surface',
        autosize=False,
        width=1000,
        height=750,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=x_name),
            yaxis=dict(title='f({})'.format(x_name)),
            zaxis=dict(title='{}R_l({},f({}))'.format('-' if flip_z else '', x_name, x_name)),
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    fig = go.Figure(data=data, layout=layout)
    ply.iplot(fig, show_link=False)


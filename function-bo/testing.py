#!/usr/bin/env python3

import numpy as np
import scipy
import GPy
import matplotlib.pyplot as plt


def in_jupyter():
    """ whether the current script is running in IPython/Jupyter """
    try:
        __IPYTHON__
    except NameError:
        return False
    return True

try:
    import plotly.offline as ply
    import plotly.graph_objs as go

    if in_jupyter():
        print('plotly setup')
        ply.init_notebook_mode(connected=False)
except ImportError:
    pass


xmin, xmax = 0, 10
ymin, ymax = -5, 5

def to_fit(x):
    return np.sin(x) * 2*np.cos(x/4)


def integrate(f, domain, intervals=100):
    """ approximate the integral of an arbitrary function defined on self.domain with
    the given number of intervals using Simpson's method
    """
    # Simpson's method approximates section areas using a quadratic polynomial
    # better approximation than trapezoid rule (which uses a straight line)
    assert intervals % 2 == 0, 'Simpson\'s rule requires an even number of intervals'
    xs = np.linspace(domain[0], domain[1], num=intervals+1)
    ys = [f(x) for x in xs]
    return scipy.integrate.simps(y=ys, x=xs)

def plot_to_fit():
    xs = np.linspace(xmin, xmax, num=100)
    plt.plot(xs, [to_fit(x) for x in xs], 'k--', label='to fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_f(f, show_to_fit=True, ax=None):
    if show_to_fit:
        print('area between: {}'.format(integrate(lambda x: abs(to_fit(x)-f(x)), (xmin, xmax))))
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.linspace(xmin, xmax, num=200)
    fs = [f(x) for x in xs]
    ax.plot(xs, fs, color='salmon', label='f')
    if show_to_fit:
        fits = [to_fit(x) for x in xs]
        ax.plot(xs, fits, linestyle='--', color='k', label='to fit')
        ax.fill_between(xs, [min(a, b) for a,b in zip(fits, fs)], [max(a,b) for a,b in zip(fits, fs)], alpha=0.2)
    ax.legend()

def plot_trials(trials, show_to_fit=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.linspace(xmin, xmax, num=200)
    if show_to_fit:
        ax.plot(xs, [to_fit(x) for x in xs], 'k--', label='to fit')
    for t in trials:
        ys = [t.f(x) for x in xs]
        ax.plot(xs, ys, color='grey', alpha=0.4)
    ax.legend()


def grid_to_points(grid):
    ''' take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix '''
    # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
    return np.vstack((grid[0].ravel(), grid[1].ravel())).T

def points_to_grid(points, grid):
    ''' take a matrix of points generated with grid_to_points and return it to a grid'''
    return points.reshape(*grid[0].shape)

def plot_surrogate(optimiser, surrogate, show_to_fit=True, show_trials=True):
    fig, ax = plt.subplots(figsize=(20, 10))

    # undo seaborn styling
    ax.grid(False)
    ax.set_facecolor((1, 1, 1))

    x_name = optimiser.bounds[0]
    xs = np.linspace(xmin, xmax, num=100)

    sampled_ys = surrogate.X[:,1]
    ymin, ymax = np.min(sampled_ys), np.max(sampled_ys)
    y_margin = (ymax-ymin)*0.1
    ys = np.linspace(ymin-y_margin, ymax+y_margin, num=100)

    grid = np.meshgrid(xs, ys)
    mus_points, var_points = surrogate.predict(grid_to_points(grid))
    mus_grid = points_to_grid(mus_points, grid)

    cmap = 'viridis' if optimiser.is_maximising() else 'viridis_r'
    im = ax.pcolormesh(grid[0], grid[1], mus_grid, cmap=cmap)
    c = fig.colorbar(im, ax=ax)
    c.set_label(r'$R_l({},f({}))$'.format(x_name, x_name))

    if show_to_fit:
        ax.plot(xs, [to_fit(x) for x in xs], 'k--', label='to fit')
    if show_trials:
        for t in optimiser.trials:
            ax.plot(xs, [t.f(x) for x in xs], color='grey', alpha=0.4)

    ax.set_xlabel('${}$'.format(x_name))
    ax.set_ylabel(r'$f({})$'.format(x_name))
    ax.legend()

def plot_surrogate_3D(optimiser, surrogate):
    x_name = optimiser.bounds[0]
    xs = np.linspace(xmin, xmax, num=100)

    sampled_ys = surrogate.X[:,1]
    ymin, ymax = np.min(sampled_ys), np.max(sampled_ys)
    y_margin = (ymax-ymin)*0.1
    ys = np.linspace(ymin-y_margin, ymax+y_margin, num=100)

    grid = np.meshgrid(xs, ys)
    mus_points, var_points = surrogate.predict(grid_to_points(grid))
    if optimiser.is_minimising():
        mus_points = -mus_points
    mus_grid = points_to_grid(mus_points, grid)

    data = [go.Surface(x=grid[0], y=grid[1], z=mus_grid, colorscale='Viridis')]
    layout = go.Layout(
        title='3D surface',
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=x_name),
            yaxis=dict(title='f({})'.format(x_name)),
            zaxis=dict(title='{}R_l({},f({}))'.format('' if optimiser.is_maximising() else '-', x_name, x_name))
        )
    )
    fig = go.Figure(data=data, layout=layout)
    ply.iplot(fig, show_link=False)


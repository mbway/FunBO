#!/usr/bin/env python3

import numpy as np
import scipy
import GPy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import function_bo as fbo
import turbo.plotting as tp


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
    ply = None


def unzip(l):
    if l:
        return list(zip(*l))
    else:
        return [[], []] # assuming 2D list

def format_list(l, precision):
    return '[{}]'.format(', '.join([('{:.' + str(precision) + 'f}').format(v) for v in l]))


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

def get_reward_cmap(values, minimising):
    # the reward stays blue for the first half then begins changing to red
    cs = [(0, 0, 1.0), (0, 0, 0.9), (1.0, 0, 0)]
    if minimising:
        cs = list(reversed(cs))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('reward_colors', cs)
    vmin, vmax = np.min(values), np.max(values)
    assert np.isscalar(vmin) and np.isscalar(vmax)
    return mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

def plot_convergence(optimiser, best_R_g):
    fig, ax = plt.subplots(figsize=(12, 8))
    xs = np.arange(len(optimiser.trials))
    ys = [t.R_g for t in optimiser.trials]
    ax.plot(xs, ys, label='$R_g$')
    if best_R_g is not None:
        ax.axhline(y=best_R_g, linestyle=':', color='grey', label='best $R_g$')
    ax.legend()
    ax.set_ylabel('$R_g$')
    ax.set_xlabel('trial')
    ax.set_title('Convergence')

def plot_trial_area(optimiser, trial, true_best, quiet=False, ax=None):
    _, xmin, xmax = optimiser.domain_bounds
    if not quiet:
        print('area between: {}'.format(integrate(lambda x: abs(true_best(x)-trial.f(x)), (xmin, xmax))))
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

def plot_trials(optimiser, trials, true_best=None, color_by_reward=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    _, xmin, xmax = optimiser.domain_bounds
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


def grid_to_points(grid):
    ''' take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix '''
    # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
    return np.vstack((grid[0].ravel(), grid[1].ravel())).T

def points_to_grid(points, grid):
    ''' take a matrix of points generated with grid_to_points and return it to a grid'''
    return points.reshape(*grid[0].shape)

def plot_acquisition(optimiser, surrogate, x, prev_xy=None, beta=1, l=1):
    fig = plt.figure(figsize=(16, 8))
    grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    ymin, ymax = optimiser.range_bounds
    for ax in (ax1, ax2):
        margin = (ymax-ymin)*0.005
        ax.set_xlim((ymin-margin, ymax+margin))

    intervals = 100

    assert len(x.shape) == 1, 'x not 0-dimensional'
    xs = np.repeat(x.reshape(1, -1), intervals, axis=0) # stack copies of x as rows
    ys = np.linspace(*optimiser.range_bounds, num=intervals)

    mu, var = surrogate.predict(np.hstack((xs, ys.reshape(-1, 1))))

    # Surrogate plot
    ax1.plot(ys, mu, label='surrogate $\mu$')
    sig = np.sqrt(var)
    n_sig = 2
    ax1.fill_between(ys, (mu-n_sig*sig).flatten(), (mu+n_sig*sig).flatten(),
                     alpha=0.2, label=r'surrogate ${}\sigma$'.format(n_sig))
    ax1.set_xlabel('$f(x)$')
    ax1.set_ylabel('$R_l(x,f(x))$')
    ax1.legend()

    # Acquisition plot
    def get_X(ys):
        xs = np.repeat(x.reshape(1, -1), ys.shape[0], axis=0) # stack copies of x as rows
        return np.hstack((xs, ys))

    def acq(ys):
        return fbo.UCB(get_X(ys), beta=beta, surrogate=surrogate, maximising=optimiser.is_maximising())
    UCBs = acq(ys.reshape(-1, 1))
    UCB_min = np.min(UCBs)
    UCBs -= UCB_min # shift to sit on x axis

    ax2.plot(ys, UCBs, color='C1', label=r'$\alpha$')
    best_y, info = fbo.maximise(acq, (ymin, ymax))
    ax2.plot(best_y, info['max_acq']-UCB_min, 'o', color='orange', label=r'$\max\,\alpha$')

    if prev_xy is not None:
        ws = fbo.tracking_weights(get_X(ys.reshape(-1, 1)), prev_xy, l=l)
        ax2.plot(ys, ws*0.4*np.max(UCBs), color='C2', label=r'$k((x_{prev},f(x_{prev})), (x,f(x)))$ (rescaled)')

        ax2.plot(ys, UCBs*ws, color='C3', label=r'$k\alpha$')
        def tracked_acq(ys):
            X = get_X(ys)
            UCBs = fbo.UCB(X, beta=beta, surrogate=surrogate, maximising=optimiser.is_maximising())
            UCBs -= np.min(UCBs) # shift to sit on x axis
            ws = fbo.tracking_weights(X, prev_xy, l=l)
            return UCBs*ws

        best_y, info = fbo.maximise(tracked_acq, (ymin, ymax))
        ax2.plot(best_y, info['max_acq'], 'ro', label=r'$\max\,k\alpha$')

    for ax in (ax1, ax2):
        ax.axvline(best_y, linestyle='--', color=(1.0, 0.3, 0.3), alpha=0.4, zorder=-1)
        prev_y = prev_xy[0,-1]
        ax.axvline(x=prev_y, linestyle=':', color='grey', alpha=0.4, zorder=-1, label='$f(x_{prev})$' if ax == ax2 else None)

    ax2.set_xlabel('$f(x)$')
    ax2.set_ylabel(r'$\alpha(f(x))$')
    ax2.legend()

    fig.tight_layout()

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

    x_name, xmin, xmax = optimiser.domain_bounds
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
        norm = tp.MidpointNorm(np.min(mus_points), np.max(mus_points), midpoint=None, midpoint_fraction=midpoint_fraction)
    im = ax.pcolormesh(grid[0], grid[1], mus_grid, cmap=cmap, norm=norm)
    c = fig.colorbar(im, ax=ax)
    c.set_label(r'$R_l({},f({}))$'.format(x_name, x_name))

    ax.set_xlabel('${}$'.format(x_name))
    ax.set_ylabel(r'$f({})$'.format(x_name))

    return fig

def plot_surrogate_3D(optimiser, surrogate, show_var=True, flip_z=False):
    assert ply is not None, 'plotly not imported'
    assert surrogate is not None, 'no surrogate'
    x_name, xmin, xmax = optimiser.domain_bounds
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


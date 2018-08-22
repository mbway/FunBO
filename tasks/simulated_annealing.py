#!/usr/bin/env python3
"""
An implementation of simulated annealing

References:
    - Locatelli, M., 2002. Simulated annealing algorithms for continuous global optimization. In Handbook of global optimization (pp. 179-229). Springer, Boston, MA.
    - Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997. Enhanced simulated annealing for globally minimizing functions of many-continuous variables. ACM Transactions on Mathematical Software (TOMS), 23(2), pp.209-228.
"""

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
from IPython.display import display, HTML

try:
    import plotly.offline as ply
    import plotly.graph_objs as go

    ply.init_notebook_mode(connected=False)
except ImportError:
    ply = None

import funbo as fb

def Metropolis_acceptance(current_E, candidate_E, temperature):
    """ Metropolis acceptance function. Returns the probability of accepting the candidate.

    - always accepts if the candidate has lower energy
    - accepts with some probability <1 if the candidate has higher energy

    The function has the value 1 from -infty to current_E then decays
    exponentially to 0, with the decay rate controlled by the temperature
    (larger => slower decay)
    """
    if candidate_E <= current_E:
        return 1
    else:
        return np.exp(-(candidate_E - current_E)/temperature)

def uniform_in_hypersphere(d, r):
    """ sample a uniform random vector from inside a hypersphere with D
    dimensions and a radius of r

    uses rejection sampling which may be inefficient
    """
    while True:
        X = np.random.uniform(-1, 1, size=d)
        if np.dot(X, X) < 1: # within a unit hypersphere (squared magnitude < 1)
            return X * r # scale to the desired radius

def uniform_on_hypersphere_surface(d, r):
    """ sample a random vector on the surface of a hypersphere, in other words a
    d-dimensional vector with length r and a random direction.

    r should be chosen so that ~60% of the steps are accepted

    Candidate distribution used in Bohachevsky et al., 1986
    """
    X = np.random.uniform(-1, 1, size=d)
    X /= np.linalg.norm(X)
    return X * r


def temperature_exponential_decay(i, M, factor, T_0, T_min):
    """ decay the temperature from T_0 at iteration i=0 by a factor of 'factor' after each group of M iterations
    Args:
        factor: (0,1) the decay factor

    Vanderbilt and Louie, 1984
    """
    return max(factor**np.floor(i/M) * T_0, T_min)

def simulated_annealing(energy_func, initial_state, max_its, candidate_dist, cooling_schedule, acceptance=Metropolis_acceptance):
    """ Simulated Annealing

    Args:
        energy_func: a function which takes a state and returns a scalar value
            for the 'energy'/'cost'/'objective value' of the state. This is the
            function to be minimised.
        initial_state: the initial state
        max_its: the maximum number of iterations to perform
        candidate_dist: a function which takes the current state, state record,
            acceptance_record and temperature and returns a new state which may
            or may not be accepted. The function should perform any bounds
            or validity checking that is required and should not only return
            a valid candidate.
        cooling_schedule: a function from iteration number (0 to max_its-1) to a temperature > 0
        acceptance: a function which takes the current energy, candidate state
            energy and the current temperature and returns the probability of
            accepting the candidate.

    Note:
        "the cooling schedule should always take into account the function to be optimized"


    Corana et al., 1987 introduced the idea of anisotropic searching by changing one dimension at a time. The maximum step in each dimension is updated based on the probability of the perturbations along the dimension being accepted, aiming to keep it at 60%
    Siarry et al., 1997 introduced a variant where a subset of the dimensions are perturbed at a time.
    """
    current_state = initial_state
    current_E = energy_func(initial_state)

    state_record = [(initial_state, current_E)]
    acceptance_record = []

    for i in range(max_its):
        temperature = cooling_schedule(i)
        candidate = candidate_dist(current_state, state_record, acceptance_record, temperature)
        candidate_E = energy_func(candidate)
        a = acceptance(current_E, candidate_E, temperature)
        if a == 1 or a >= np.random.uniform(0, 1): # accept
            current_state = candidate
            current_E = candidate_E
            state_record.append((current_state, current_E))
            acceptance_record.append(True)
        else:
            acceptance_record.append(False)

    best_state, best_E = min(state_record, key=lambda x: x[1])
    return best_state, best_E, state_record


def elastic_net_simulated_annealing(net, fit_cost, initial_amplitudes, max_its,
                                    CP_batch_size, candidate_func,
                                    cooling_schedule,
                                    acceptance=Metropolis_acceptance,
                                    record_state=True):
    # current values
    c_amplitudes = initial_amplitudes
    c_fit_costs = fit_cost(net, c_amplitudes)
    c_E = np.sum(c_fit_costs) + np.sum(net.elastic_potentials(c_amplitudes))

    best_amplitudes = c_amplitudes
    best_E = c_E

    indices = np.vstack(net.grid.get_indices())
    num_CPs = indices.shape[0]
    batch_its = len(indices)//CP_batch_size + 1 # number of batches in a single iteration
    acceptance_record = np.zeros(shape=(max_its * batch_its,))
    if record_state:
        state_record = [(c_amplitudes, c_E)]

    for i in range(max_its):
        temperature = cooling_schedule(i)
        # new values
        n_amplitudes = candidate_func(c_amplitudes)
        n_fit_costs = fit_cost(net, n_amplitudes)
        # working values, updated each batch
        w_amplitudes = c_amplitudes.copy()
        w_fit_costs = c_fit_costs.copy()
        w_E = c_E

        # to introduce some isotropy, apply small batches of perturbations at a time, leaving the rest of the net the same

        batches = np.array_split(np.random.permutation(num_CPs), num_CPs//CP_batch_size)
        for batch_i, b in enumerate(batches):
            # ndarrays can be indexed with X[(x1, x2, ...), (y1, y2, ...)]
            # not with X[(x1,y1), (x2,y2), ...] as I originally thought
            batch_indices = tuple(tuple(idx) for idx in indices[b].T)
            # assume the batch will be accepted and roll back if it is not
            w_amplitudes[batch_indices] = n_amplitudes[batch_indices]
            w_fit_costs[batch_indices] = n_fit_costs[batch_indices]
            # elastic costs are effect neighbouring points as well, so easier to
            # just recalculate the whole thing.
            #TODO: look into updating each point in the batch and its neighbours rather than recalculating the whole thing if it is too slow. Also look into ways of making the grid faster overall if this is an issue. Perhaps look into jit?
            w_elastic_costs = net.elastic_potentials(w_amplitudes)

            batch_E = np.sum(w_fit_costs) + np.sum(w_elastic_costs)
            a = acceptance(w_E, batch_E, temperature)
            if a == 1 or a >= np.random.uniform(0, 1):
                # accepted
                w_E = batch_E
                acceptance_record[i*batch_its + batch_i] = 1 # record that this batch was accepted
            else:
                # rejected, roll back
                w_amplitudes[batch_indices] = c_amplitudes[batch_indices]
                w_fit_costs[batch_indices] = c_fit_costs[batch_indices]
        # all batches considered, set the working amplitudes as current
        c_amplitudes = w_amplitudes
        c_fit_costs = w_fit_costs
        c_E = w_E
        if w_E < best_E:
            best_amplitudes = w_amplitudes.copy()
            best_E = w_E
        if record_state:
            state_record.append((w_amplitudes.copy(), w_E))

    if record_state:
        return best_amplitudes, best_E, state_record, acceptance_record
    else:
        return best_amplitudes, best_E





def plot_elastic_net(net, amplitudes, tooltips=None, goal_XYZ=None, axes_names=('x','y','z'), axes_limits=(None,None,None)):
    '''plot a 3D surface using plotly

    Parameters should be of the form:
    ```
    X = np.arange(...)
    Y = np.arange(...)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)
    ```

    Args:
        tooltips: an array with the same length as the number of points,
            containing a string to display beside them
    '''
    netX, netY = net.grid.meshgrid(cartesian_index=False)

    # any transparency causes strange coloring glitches
    mesh_surface = go.Surface(
        x=netX, y=netY, z=amplitudes,
        text=tooltips, colorscale='Red', opacity=1, showscale=False
    )
    mesh_scatter = go.Scatter3d(
        x=netX.flatten(), y=netY.flatten(), z=amplitudes.flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            opacity=1
    ))

    data = [mesh_surface, mesh_scatter]

    if goal_XYZ is not None:
        goal_surface = go.Surface(
            x=goal_XYZ[0], y=goal_XYZ[1], z=goal_XYZ[2],
            text=tooltips, colorscale='Viridis', opacity=1
        )
        data.append(goal_surface)

    layout = go.Layout(
        title='3D Plot',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=axes_names[0], range=axes_limits[0]),
            yaxis=dict(title=axes_names[1], range=axes_limits[1]),
            zaxis=dict(title=axes_names[2], range=axes_limits[2])
        )
    )
    fig = go.Figure(data=data, layout=layout)
    # show_link is a link to export to the 'plotly cloud'
    ply.iplot(fig, show_link=False)


# https://plot.ly/python/visualizing-mri-volume-slices/


def plot_elastic_net_animation_interactive(net, amplitudes, tooltips=None,
                               goal_XYZ=None, axes_names=('x','y','z'), axes_limits=(None,None,None),
                               frame_duration=200):
    '''plot a 3D surface using plotly

    Parameters should be of the form:
    ```
    X = np.arange(...)
    Y = np.arange(...)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)
    ```

    Args:
        tooltips: an array with the same length as the number of points,
            containing a string to display beside them
    '''
    netX, netY = net.grid.meshgrid(cartesian_index=False)
    netX_flat, netY_flat = netX.flatten(), netY.flatten()
    netZs = amplitudes
    elasticZs = [net.elastic_potentials(a) for a in amplitudes]

    # showscale = whether to show the colorbar

    if goal_XYZ is not None:
        goal_surface = dict(x=goal_XYZ[0], y=goal_XYZ[1], z=goal_XYZ[2],
                            colorscale='Viridis', opacity=0.8, showscale=False, type='surface')

    frames = []
    for i, netZ in enumerate(netZs):
        data = []
        if goal_XYZ is not None:
            data.append(goal_surface)
        elastics = elasticZs[i]
        # any transparency causes strange coloring glitches
        net_surface = dict(x=netX, y=netY, z=netZ, type='surface',
                           colorscale='Red', surfacecolor=elastics,
                           cmin=np.min(elastics), cmax=np.max(elastics),
                           showscale=False, opacity=1)
        net_scatter = dict(x=netX_flat, y=netY_flat, z=amplitudes[i].flatten(), type='scatter3d',
                           mode='markers', marker=dict(size=2, color='black'))

        net_lines = []
        line_style=dict(width=1.5, color='black')
        for a, b, c in zip(netX, netY, amplitudes[i]):
            net_lines.append(dict(x=a, y=b, z=c, type='scatter3d', mode='lines', line=line_style))
        for a, b, c in zip(netX.T, netY.T, amplitudes[i].T):
            net_lines.append(dict(x=a, y=b, z=c, type='scatter3d', mode='lines', line=line_style))

        data += [net_surface, net_scatter] + net_lines
        frame = dict(data=data, name=str(i))
        frames.append(frame)
    data = frames[0]['data']

    play_button = dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=frame_duration, redraw=False))])

    slider = dict(steps=[],
        transition=dict(duration=0),
        x=0,# initial value
        y=0,
        currentvalue=dict(prefix='iteration: ', visible=True),
        len=1.0# slider length
    )
    for i in range(len(frames)):
        name = str(i)
        step = dict(method='animate', label=name,
                    args=[[name],
                        dict(mode='immidiate',
                             frame=dict(duration=100, redraw=False),
                             transition=dict(duration=0)
                        )
                    ]
        )
        slider['steps'].append(step)

    layout = dict(
        title='3D Plot',
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=axes_names[0], range=axes_limits[0]),
            yaxis=dict(title=axes_names[1], range=axes_limits[1]),
            zaxis=dict(title=axes_names[2], range=axes_limits[2])
        ),
        showlegend=False,
        updatemenus=[dict(type='buttons', buttons=[play_button])],
        sliders=[slider]
    )
    fig = dict(data=data, layout=layout, frames=frames)
    # show_link is a link to export to the 'plotly cloud'
    ply.iplot(fig, show_link=False)


def plot_elastic_net_animation(net, amplitudes, goal_XYZ=None,
                               axes_names=('x','y','z'),
                               axes_limits=(None,None,None),
                               frame_duration=200, display=True):
    """
    """

    netX, netY = net.grid.meshgrid(cartesian_index=False)
    netX_flat, netY_flat = netX.flatten(), netY.flatten()
    netZs = amplitudes
    elasticZs = [net.elastic_potentials(a) for a in amplitudes]

    # showscale = whether to show the colorbar

    fig = plt.figure(figsize=(14, 8))
    ax = fig.gca(projection='3d')
    fig.tight_layout()

    def update(frame):
        print('#', end='')
        ax.cla()
        frame = -1 if frame >= len(netZs) else frame
        e = elasticZs[frame]
        cmin, cmax = np.min(e), np.max(e)
        e = (e - cmin) / (cmax-cmin)
        surf = ax.plot_surface(netX, netY, netZs[frame], rstride=1, cstride=1,
                               facecolors=mpl.cm.Reds(e), alpha=0.8,
                               edgecolors='black', linewidth=1,
                               antialiased=True)
        if axes_limits[0] is not None: ax.set_xlim(axes_limits[0])
        if axes_limits[1] is not None: ax.set_ylim(axes_limits[1])
        if axes_limits[2] is not None: ax.set_zlim(axes_limits[2])
        ax.margins(0)
        return surf,

    ani = mpl.animation.FuncAnimation(fig, update, frames=np.arange(len(netZs) + 5), interval=frame_duration)
    if display:
        display_video(ani)
        plt.close(fig)
    else:
        return ani


def save_video(ani, filename, fps=10):
    ffmpegWriter = mpl.animation.writers['ffmpeg']
    w = ffmpegWriter(fps=fps, metadata=dict(), bitrate=1800)
    ani.save(filename, writer=w)

def display_video(ani):
    display(HTML(ani.to_html5_video()))


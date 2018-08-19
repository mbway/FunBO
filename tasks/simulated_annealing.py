#!/usr/bin/env python3
"""
An implementation of simulated annealing

References:
    - Locatelli, M., 2002. Simulated annealing algorithms for continuous global optimization. In Handbook of global optimization (pp. 179-229). Springer, Boston, MA.
    - Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997. Enhanced simulated annealing for globally minimizing functions of many-continuous variables. ACM Transactions on Mathematical Software (TOMS), 23(2), pp.209-228.
"""

import numpy as np

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


def temperature_exponential_decay(i, M, factor, T_0):
    """ decay the temperature from T_0 at iteration i=0 by a factor of 'factor' after each group of M iterations
    Args:
        factor: (0,1) the decay factor

    Vanderbilt and Louie, 1984
    """
    return factor**np.floor(i/M) * T_0

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


class ElasticNet:
    """
    A class for managing an elastic net which is defined by a D-dimensional grid
    of 'control point' locations and a tensor of amplitudes at each control
    point location.
    """
    def __init__(self, grid, elastic_stiffness, range_bounds):
        self.grid = grid
        self.elastic_stiffness = elastic_stiffness
        self.range_bounds = range_bounds

    def get_interpolated_fun(self, amplitudes):
        assert amplitudes.shape == self.grid.num_values
        return fb.utils.InterpolatedFunction(self.grid, amplitudes.reshape(-1, 1), clip_range=self.range_bounds)

    def random_amplitudes(self):
        return np.random.uniform(*self.range_bounds, size=self.grid.num_values)

def elastic_net_simulated_annealing(net):
    pass


def plot_elastic_net(grid, net, amplitude, tooltips=None, goal_z=None, axes_names=('x','y','z')):
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
    X, Y = grid.meshgrid(cartesian_index=False)

    netX, netY = net.grid.meshgrid(cartesian_index=False)
    netZ = grid.fun_on_grid(net.get_interpolated_fun(amplitude))

    mesh_surface = go.Surface(
        x=X, y=Y, z=netZ,
        text=tooltips, colorscale='Red', opacity=0.8
    )
    mesh_scatter = go.Scatter3d(
        x=netX.flatten(), y=netY.flatten(), z=amplitude.flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            opacity=1
    ))

    data = [mesh_surface, mesh_scatter]

    if goal_z is not None:
        goal_surface = go.Surface(
            x=X, y=Y, z=goal_z,
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
            xaxis=dict(title=axes_names[0]),
            yaxis=dict(title=axes_names[1]),
            zaxis=dict(title=axes_names[2])
        )
    )
    fig = go.Figure(data=data, layout=layout)
    # show_link is a link to export to the 'plotly cloud'
    ply.iplot(fig, show_link=False)


# https://plot.ly/python/visualizing-mri-volume-slices/


def plot_elastic_net_animation(grid, net, amplitudes, tooltips=None,
                               goal_Z=None, axes_names=('x','y','z'), axes_limits=(None,None,None),
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
    X, Y = grid.meshgrid(cartesian_index=False)

    netX, netY = net.grid.meshgrid(cartesian_index=False)
    netX, netY = netX.flatten(), netY.flatten()
    netZs = [grid.fun_on_grid(net.get_interpolated_fun(a)) for a in amplitudes]

    # showscale = whether to show the colorbar

    if goal_Z is not None:
        goal_surface = dict(x=X, y=Y, z=goal_Z,
                            colorscale='Viridis', opacity=1, showscale=False, type='surface')

    frames = []
    for i, netZ in enumerate(netZs):
        data = []
        if goal_Z is not None:
            data.append(goal_surface)
        net_surface = dict(x=X, y=Y, z=netZ, type='surface', colorscale='Red', showscale=False, opacity=0.8)
        net_scatter = dict(x=netX, y=netY, z=amplitudes[i].flatten(), type='scatter3d',
                           mode='markers', marker=dict(size=2, color='black'))
        data += [net_surface, net_scatter]
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
        width=900,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=axes_names[0], range=axes_limits[0]),
            yaxis=dict(title=axes_names[1], range=axes_limits[1]),
            zaxis=dict(title=axes_names[2], range=axes_limits[2])
        ),
        updatemenus=[dict(type='buttons', buttons=[play_button])],
        sliders=[slider]
    )
    fig = dict(data=data, layout=layout, frames=frames)
    # show_link is a link to export to the 'plotly cloud'
    ply.iplot(fig, show_link=False)


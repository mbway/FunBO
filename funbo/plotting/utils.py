#!/usr/bin/env python3
"""
Utility functions used for plotting
"""

import warnings
import numpy as np
import matplotlib as mpl
import scipy.interpolate

from funbo.utils import show_warnings


def in_jupyter():
    """ whether the current script is running in IPython/Jupyter """
    try:
        __IPYTHON__
    except NameError:
        return False
    return True


def integrate(f, domain, intervals=100):
    """ approximate the integral of an arbitrary function defined on `domain` with
    the given number of intervals using Simpson's method
    """
    # Simpson's method approximates section areas using a quadratic polynomial
    # better approximation than trapezoid rule (which uses a straight line)
    assert intervals % 2 == 0, 'Simpson\'s rule requires an even number of intervals'
    xs = np.linspace(domain[0], domain[1], num=intervals+1)
    ys = [f(x) for x in xs]
    return scipy.integrate.simps(y=ys, x=xs)

def multidimensional_integrate(f, domain, quiet=True):
    with warnings.catch_warnings(record=True) as ws:
        result, abserr = scipy.integrate.nquad(func=f, ranges=domain, opts=dict(limit=80, epsabs=1e-4, epsrel=1e-4))
    if not quiet:
        show_warnings(ws)
    return result



def unzip(l):
    if l:
        return list(zip(*l))
    else:
        return [[], []] # assuming 2D list

def format_list(l, precision):
    return '[{}]'.format(', '.join([('{:.' + str(precision) + 'f}').format(v) for v in l]))



def grid_to_points(grid):
    ''' take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix '''
    # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
    return np.vstack((grid[0].ravel(), grid[1].ravel())).T

def points_to_grid(points, grid):
    ''' take a matrix of points generated with grid_to_points and return it to a grid'''
    return points.reshape(*grid[0].shape)

def get_reward_cmap(values, minimising):
    # the reward stays blue for the first half then begins changing to red
    cs = [(0, 0, 1.0), (0, 0, 0.9), (1.0, 0, 0)]
    if minimising:
        cs = list(reversed(cs))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('reward_colors', cs)
    vmin, vmax = np.min(values), np.max(values)
    assert np.isscalar(vmin) and np.isscalar(vmax)
    return mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

def integer_ticks(axis):
    """
    Args:
        axis: ax.xaxis or ax.yaxis
    """
    axis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))


# from turbo
class MidpointNorm(mpl.colors.Normalize):
    '''Warp the colormap so that more of the available colors are used on the range of interesting data.

    Half of the color map is used for values which fall below the midpoint,
    and half are used for values which fall above.
    This can be used to draw attention to smaller differences at the extreme
    ends of the observed values.

    based on:
        - http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
        - https://matplotlib.org/users/colormapnorms.html
    '''
    def __init__(self, vmin, vmax, midpoint, midpoint_fraction=None, res=100, clip=False):
        '''
        Args:
            vmin: the minimum possible z/height value
            vmax: the maximum possible z/height value
            midpoint: the value to 'center' around
            midpoint_fraction: if not None, overrides midpoint and sets the
                midpoint as a fraction of the distance between vmin and vmax. eg
                0.75 => three quarters of the way to vmax.
            res: the 'resolution' ie number of distinct levels in the colorbar
            clip: whether to clip the z values to [0,1] if they lie outside [vmin, vmax]

        Note: according to `mpl.colors.Normalize` documentation: If vmin or vmax
            is not given, they are initialized from the minimum and maximum
            value respectively of the first input processed.
        '''
        super().__init__(vmin, vmax, clip)
        if midpoint_fraction is None:
            self.midpoint = midpoint
        else:
            self.midpoint = vmin + (vmax-vmin)*midpoint_fraction
        self.res = res

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    def levels(self):
        '''
        Returns:
            a numpy array for the values where the boundaries between the colors
            should be placed.
        '''
        return np.concatenate((
            np.linspace(self.vmin, self.midpoint, num=self.res/2, endpoint=False),
            np.linspace(self.midpoint, self.vmax, num=self.res/2)))


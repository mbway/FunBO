#!/usr/bin/env python3
"""
Utility functions and data structures
"""

import warnings
import numpy as np
import scipy.interpolate
import time

from .grid import Grid

class FixedAttributes:
    """ Prevent assignment to attributes other than those defined in the class
        attribute `slots`.

    For several classes I would like to prevent assignment to arbitrary
    attributes to catch typos and errors caused by change to the API during
    development. Once development has settled, using __slots__ instead would be
    slightly beneficial.

    This class is a work-around for an unfortunate bug with IPython/Jupyter
    where classes using __slots__ do not automatically re-load and so when an
    underlying source file changes the objects in that session become useless.
    Every attempt to access an attribute gives the error:

        `TypeError: descriptor '<ATTR>' for '<CLASS>' objects doesn't apply to '<CLASS>' object`

    Because IPython doesn't understand that the class hasn't changed since the re-load.

    This class superficially performs the same job as __slots__, but without any
    of the performance or space gains.
    """
    def __setattr__(self, name, value):
        if name in self.slots:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('{} Object does not have an "{}" attribute!'.format(type(self), name))

    def __init__(self, *args):
        assert len(args) == len(self.slots), 'not enough arguments'
        for i, a in enumerate(args):
            setattr(self, self.slots[i], a)

    def _null_init(self):
        """ initialise all slots to None """
        FixedAttributes.__init__(self, *[None]*len(self.slots))

    def __repr__(self):
        attrs = ', '.join([a + '=' + repr(getattr(self, a)) for a in self.slots])
        return '{}({})'.format(type(self).__name__, attrs)


class InterpolatedFunction:
    def __init__(self, control_xs, control_ys, interpolation='linear', clip_range=None):
        """
        Args:
            control_xs: a Grid or an (N,A) array of control point locations as rows
            control_ys: a (N,1) array of values corresponding to a function value at `control_xs`
            interpolation: the interpolation method to use (see scipy.interpolate.griddata)
            clip_range: (min, max) restrict the output of the interpolated
                function to lie within the given range (by clamping the output
                value), or None to allow any output value.
        """
        if isinstance(control_xs, Grid):
            self.control_xs = control_xs.get_points()
        else:
            assert len(control_xs.shape) == 2 and control_xs.shape[0] > 1
            self.control_xs = control_xs
        assert control_ys.shape == (self.control_xs.shape[0], 1)
        self.control_ys = control_ys
        self.interpolation = interpolation
        # note: control_ys may be outside of clip_range however it is probably
        # better to just leave that and only clip the function output.
        self.clip_range = clip_range

        # more interpolation options in 1D. Construct the interpolated function
        # once here rather than every evaluation.
        # implementation similar to scipy.interpolate.griddata, except that
        # function constructs the interpolated function for every query.
        if self.control_xs.shape[1] == 1: # one dimensional
            # additional interpolation methods are supported in one dimension
            # bounds_error=False to instead fill values with nan, which is then detected afterwards
            # note: requires self.control_xs to be sorted, which Grid already ensures
            self.f = scipy.interpolate.interp1d(self.control_xs.flatten(), self.control_ys.flatten(), kind=self.interpolation, bounds_error=False, fill_value=np.nan)

        elif isinstance(control_xs, Grid) and self.interpolation in ('linear', 'nearest'):
            # if the points are defined on a grid (even with uneven spacing) then this interpolation will be applicable and more efficient
            if any(d != 1 for d in control_xs.traverse_directions) or control_xs.endianness != 'big':
                raise NotImplementedError('control_xs must be a big endian grid with all +1 traverse directions')
            # see Grid.fun_on_grid
            ys = self.control_ys.reshape(control_xs.shape)
            self.f = scipy.interpolate.RegularGridInterpolator(control_xs.values, ys, method=self.interpolation, bounds_error=False, fill_value=np.nan)
        else:
            # 'unstructured data' interpolation methods
            warnings.warn('warning: since control_xs is not defined on a regular grid, using less efficient interpolation methods.')

            if self.interpolation == 'nearest':
                self.f = scipy.interpolate.NearestNDInterpolator(self.control_xs, self.control_ys, bounds_error=False, fill_value=np.nan, rescale=False)
            elif self.interpolation == 'linear':
                self.f = scipy.interpolate.LinearNDInterpolator(self.control_xs, self.control_ys, fill_value=np.nan, rescale=False)
            elif self.interpolation == 'cubic' and self.control_xs.shape[1] == 2: # two dimensional
                self.f = scipy.interpolate.CloughTocher2DInterpolator(self.control_xs, self.control_ys, fill_value=np.nan, rescale=False)
            elif self.interpolation == 'rbf':
                # scipy.interpolate.Rbf
                raise NotImplemented() # might be interesting
            else:
                raise ValueError()

    def __call__(self, X):
        """ Evaluate the interpolated function at the given points

        Args:
            X: the points (as rows) to evaluate at. If the function one
                dimensional then X may also be a scalar.

        Returns:
            the interpolated function values corresponding to X. If X is a
            single point then the result is a scalar, otherwise an (N,1) array
        """
        if self.control_xs.shape[1] == 1: # one dimensional
            X = np.array([[X]]) if np.isscalar(X) else np.asarray(X)
        else:
            X = np.asarray(X)
        assert len(X.shape) == 2 and X.shape[1] == self.control_xs.shape[1]

        Y = self.f(X).reshape(-1, 1)

        nans = X[np.argwhere(np.isnan(Y.flatten()))]
        if nans.size > 0:
            raise ValueError('InterpolatedFunction evaluated at the following point(s) outside of the interpolation range:\n{}'.format(nans))

        if self.clip_range is not None:
            Y = np.clip(Y, *self.clip_range)

        return np.asscalar(Y) if Y.size == 1 else Y



def uniform_random_in_bounds(num_samples, bounds):
    """ generate `num_samples` uniform randomly distributed points inside the given bounds
    """
    assert num_samples > 0 and bounds
    # build up the vectors element by element
    cols = []
    for rmin, rmax in bounds:
        assert rmin <= rmax
        cols.append(np.random.uniform(rmin, rmax, size=(num_samples, 1)))
    return np.hstack(cols)



def k_RBF(X, center, sigma, l, return_gradient=False):
    """ calculate the RBF values of X with a given center, standard deviation and lengthscale

    Args:
        X: the points to evaluate at (one per row)
        center: the center of the RBF function
        sigma: the standard deviation of the RBF function
        l: the lengthscale of the RBF function
        return_gradient: whether to calculate and return drbf_dx
    """
    assert np.isscalar(sigma) and np.isscalar(l)
    assert len(X.shape) == 2
    assert center.shape == (1, X.shape[1])
    diff = X - center # subtract from every row
    rs = np.linalg.norm(diff, axis=1).reshape(-1, 1) # calculate the norm of every row
    rbf = sigma**2 * np.exp(-0.5 * np.square(rs/l))
    if return_gradient:
        # take sigma^2 out
        # d/dx e^x = e^x
        # chain rule so multiply by d/dx -1/(2l^2) * (x-center)^2  =  -1/l^2 * (x-center)
        drbf_dx = -1/(l**2) * diff * rbf
        return rbf, drbf_dx
    else:
        return rbf

def show_warnings(ws):
    for w in ws:
        warnings.showwarning(w.message, w.category, w.filename, w.lineno)

class Timer:
    """ A small utility for accurately timing sections of code with the minimal
    amount of clutter. Simply instantiate then call `stop()` to get the elapsed
    duration in seconds.
    """
    def __init__(self):
        self.start = time.perf_counter()
    def stop(self):
        return time.perf_counter() - self.start

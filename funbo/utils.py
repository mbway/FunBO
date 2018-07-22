#!/usr/bin/env python3
"""
Utility functions and data structures
"""

import numpy as np
import scipy.interpolate

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


#TODO: can't handle higher dimensions
#for n dimensions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
# scipy.interpolate.griddata
class PiecewiseFunction:
    def __init__(self, xs, ys, interpolation='linear', clip_range=None):
        assert len(xs) == len(ys) and len(xs) > 1
        self.control_xs = xs
        self.control_ys = ys
        self.clip_range = clip_range
        self.f = scipy.interpolate.interp1d(xs, ys, kind=interpolation)

    def __str__(self):
        cs = ', '.join(['({:.2f}, {:.2f})'.format(x, y) for x, y in zip(self.control_xs, self.control_ys)])
        return 'PiecewiseFunction[{}]'.format(cs)

    def __call__(self, x):
        try:
            if self.clip_range is not None:
                return np.clip(self.f(x), *self.clip_range)
            else:
                return self.f(x)
        except ValueError as e:
            # show the input which caused the crash
            e.args = ((e.args[0] + '  x = {}'.format(x)),)
            raise e

class InterpolatedFunction:
    pass # TODO

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

def k_RBF(r, sigma, l):
    """ calculate the RBF values of r with the given standard deviation of sigma and lengthscale of l

    Args:
        r: either a scalar length or a column of lengths
    """
    assert np.isscalar(sigma) and np.isscalar(l)
    assert np.isscalar(r) or r.shape[1] == 1 # either a scalar or a column of values
    return sigma**2 * np.exp(-0.5 * np.square(r/l))


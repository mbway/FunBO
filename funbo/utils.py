#!/usr/bin/env python3
"""
Utility functions and data structures
"""


class FixedAttributes:
    """
    A mixin class to help prevent typos by raising an error if a new attribute
    is assigned after the `immutable` flag is set in the constructor.
    """
    def __init__(self):
        self.fixed_attributes = False

    def __setattr__(self, name, value):
        """ limit the ability to set attributes. New attributes can only be
        created from the constructor.

        This prevents typos or changes to the API from failing silently.
        """
        fixed_attributes = hasattr(self, 'fixed_attributes') and self.fixed_attributes
        if hasattr(self, name) or not fixed_attributes:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('{} Object does not have an "{}" attribute!'.format(type(self), name))


#TODO: can't handle higher dimensions
#for n dimensions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
# scipy.interpolate.griddata
class PiecewiseFunction:
    def __init__(self, xs, ys, interpolation='linear', clip_range=None):
        self.control_xs = xs
        self.control_ys = ys
        self.clip_range = clip_range
        self.f = scipy.interpolate.interp1d(xs, ys, kind=interpolation)

    def __str__(self):
        cs = ', '.join(['({:.2f}, {:.2f})'.format(x, y) for x, y in zip(self.control_xs, self.control_ys)])
        return 'PiecewiseFunction{{}}'.format(cs)

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

#!/usr/bin/env python3
"""
Utility functions and data structures
"""

import warnings
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
            control_xs = control_xs.get_points()
        assert len(control_xs.shape) == 2 and control_xs.shape[0] > 1
        assert control_ys.shape == (control_xs.shape[0], 1)
        self.control_xs = control_xs
        self.control_ys = control_ys
        self.interpolation = interpolation
        self.clip_range = clip_range
        if clip_range is not None:
            assert all(clip_range[0] <= y <= clip_range[1] for y in self.control_ys)

        # more interpolation options in 1D. Construct the interpolated function
        # once here rather than every evaluation.
        if self.control_xs.shape[1] == 1: # one dimensional
            # additional interpolation methods are supported in one dimension
            # bounds_error=False to instead fill values with nan, which is then detected afterwards
            self.f1D = scipy.interpolate.interp1d(self.control_xs.flatten(), self.control_ys.flatten(), kind=self.interpolation, bounds_error=False)

    def __call__(self, X):
        """ Evaluate the interpolated function at the given points

        Args:
            X: the points (as rows) to evaluate at. If the function one
                dimensional then X may also be a scalar.

        Returns:
            the interpolated function values corresponding to X. If X is a
            single point then the result is a scalar, otherwise an (N,1) array
        """
        one_dim = self.control_xs.shape[1] == 1 # some things handled specially in one dimension
        if one_dim and np.isscalar(X):
            X = np.array([[X]])
        else:
            X = np.asarray(X)
        assert len(X.shape) == 2 and X.shape[1] == self.control_xs.shape[1]

        if one_dim:
            # additional interpolation methods are supported in one dimension
            Y = self.f1D(X.flatten()).reshape(-1, 1)
        else:
            Y = scipy.interpolate.griddata(self.control_xs, self.control_ys, X, method=self.interpolation, rescale=False)

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


def grid_to_points(grid):
    ''' take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix '''
    # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
    return np.vstack((grid[0].ravel(), grid[1].ravel())).T

def points_to_grid(points, grid_shape):
    ''' take a matrix of points generated with grid_to_points and return it to a grid'''
    return points.reshape(*grid_shape)

def fill_meshgrid(grids, f):
    G = np.empty_like(grids[0])
    for index, _ in np.ndenumerate(grids[0]):
        G[index] = f(*[g[index] for g in grids])
    return G


class Grid:
    """ an N-dimensional grid of points arranged on a grid with a discrete set
    of possible values along each dimension. The grid can be iterated over in
    the order specified by `traverse_directions`.
    """

    def __init__(self, values, traverse_directions=None):
        """
        Args:
            values: a list where each element is a list of possible values along
                that dimension.
            traverse_directions: a list of either -1 or 1 to indicate the
                direction that that dimension should be traversed when iterating
                over the grid. 1 => ascending, -1 => descending.
        """
        # the possible values along each dimension
        self.values = values
        self.dims = len(values)
        self.num_values = [len(vs) for vs in values]
        self.num_points = np.prod(self.num_values)
        self.traverse_directions = [1]*self.dims if traverse_directions is None else traverse_directions
        assert self.dims > 0
        assert all(n > 0 for n in self.num_values)
        assert len(self.traverse_directions) == self.dims
        assert all([d in (-1, 1) for d in self.traverse_directions])

    class Iterator:
        def __init__(self, grid):
            self.grid = grid
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.n < self.grid.num_points:
                p = self.grid.get_nth_point(self.n)
                self.n += 1
                return p
            else:
                raise StopIteration()

    def __iter__(self):
        # don't want to keep the iterator state in the grid object, so create an iterator
        return Grid.Iterator(self)

    def __len__(self):
        return self.num_points

    def _get_nth_coord(self, n):
        """ get the nth coordinate in the grid

        Returns:
            a coordinate into the grid, with each element being an integer index
            into the list of possible values along that dimension.
        """
        # traverse the grid with each dimension acting like a digit in a number
        assert 0 <= n < self.num_points
        coord = []
        for n_vals, direction in zip(reversed(self.num_values), reversed(self.traverse_directions)):
            n, i = divmod(n, n_vals)
            if direction == -1:
                i = n_vals - i - 1 # if tracking from the larger end, index into c from that end
            coord.insert(0, i) # prepend because building the coordinate from the 'least significant digit'
        assert n == 0
        return coord

    def _get_point(self, coord):
        """ get the point corresponding to the given grid coordinate

        Args:
            coord: a coordinate into the grid, with each element being an
                integer index into the list of possible values along that dimension.
        """
        assert len(coord) == self.dims
        return np.array([[self.values[d][coord[d]] for d in range(self.dims)]])

    def get_nth_point(self, n):
        """ get the nth point in the grid """
        return self._get_point(self._get_nth_coord(n))

    def get_points(self):
        """ get all the points in the grid as rows """
        return np.vstack(self.get_nth_point(n) for n in range(self.num_points))

    def fixed_subgrid(self, point):
        """ create a subgrid by fix the values along some dimensions while leaving others free

        Args:
            point: a list where elements are either `None` to indicate 'leave
                this dimension free' or a value `!= None` to indicate 'fix this
                dimension to this value'. The fixed values do not necessarily
                have to lie on the grid.

        Returns:
            a subgrid with the specified dimensions fixed
        """
        assert len(point) == self.dims
        values = [vals if point[d] is None else [point[d]] for d, vals in enumerate(self.values)]
        return Grid(values, self.traverse_directions)


    def subgrid(self, keep_dims):
        """ construct a grid by taking a subset of the dimensions of this grid

        Useful for plotting some dimensions while leaving most at a fixed value

        Args:
            keep_dims: a set or list of dimension _numbers_ (i.e. indices into
                self.values) to keep.
        """
        assert len(keep_dims) > 0 and all(0 <= d < self.dims for d in keep_dims)
        values = [vals for d, vals in enumerate(self.values) if d in keep_dims]
        traverse_directions = [direction for d, direction in enumerate(self.traverse_directions) if d in keep_dims]
        return Grid(values, traverse_directions)

    def meshgrid(self, cartesian_index=True):
        """ return a list of grids like the ones created by `np.meshgrid`

        useful for plotting. May want to first use `subgrid` to extract a 2D
        grid to convert to a meshgrid.

        Args:
            cartesian_index: the default indexing with `meshgrid` is 'xy' or
                'Cartesian' indexing. This causes indexing like [y,x,z,...] rather
                than 'ij' or 'matrix' indexing which causes indexing like
                [x,y,z,...]. Use Cartesian indexing for plotting.
        """
        if cartesian_index and self.dims > 1:
            # Cartesian indexing, corresponds to meshgrid(..., indexing='xy')
            # basically, the x and y axes are swapped, with the rest remaining the same
            n = np.array(self.num_values)
            n[[0,1]] = n[[1,0]] # swap elements 0 and 1
            ids = np.indices(n)
            ids[[0,1]] = ids[[1,0]] # swap elements 0 and 1
        else:
            # matrix indexing, corresponds to meshgrid(..., indexing='ij')
            ids = np.indices(self.num_values)

        assert ids.shape[0] == self.dims
        grids = []
        for d in range(self.dims):
            d_ids = ids[d]
            t = np.array(self.values[d]).dtype # copy the type of the values along this dimension
            g = np.empty_like(d_ids, dtype=t)
            for index, i in np.ndenumerate(d_ids):
                g[index] = self.values[d][i]
            grids.append(g)
        return grids


class RegularGrid(Grid):
    def __init__(self, num_values, bounds, traverse_directions=None):
        """
        Args:
            num_values: a list of the number of possible values in each
                dimension or a single scalar which applies to all dimensions
            bounds: a list of (min, max) for each dimension
            traverse_directions: a list of either -1 or 1 to indicate the
                direction that that dimension should be traversed when iterating
                over the grid. 1 => ascending, -1 => descending.
        """
        num_values = [num_values] * len(bounds) if np.isscalar(num_values) else num_values
        assert len(num_values) == len(bounds)
        # the possible values along each dimension
        values = [np.linspace(rmin, rmax, num=n) for n, (rmin, rmax) in zip(num_values, bounds)]
        super().__init__(values, traverse_directions)



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

#!/usr/bin/env python3
"""
N dimensional Grid and Elastic Net Data structures and related utilities.
"""

import warnings
import numpy as np
import scipy.interpolate
import scipy.ndimage
import time


class Grid:
    """ an N-dimensional grid of points arranged on a grid with a discrete set
    of possible values along each dimension. The grid can be iterated over in
    the order specified by `traverse_directions`.
    """

    def __init__(self, values, traverse_directions=None, traverse_order='big'):
        """
        Args:
            values: a list where each element is a list of possible values along
                that dimension.
            traverse_directions: a list of either -1 or 1 to indicate the
                direction that that dimension should be traversed when iterating
                over the grid. 1 => ascending, -1 => descending.
            traverse_order: a list of dimension indices to designate which
                dimensions are the 'least significant' and which are the 'most
                significant', to use the analogy with digits of a number.
                Earlier in the list => less significant.
                special values:
                'little' => range(dims) => little endian (first dimension is least significant)
                'big' => reversed(range(dims)) => big endian (first dimension is most significant)
        """
        # the possible values along each dimension
        self.values = values
        self.dims = len(values)
        self.shape = tuple(len(vs) for vs in values)
        self.num_points = np.prod(self.shape)
        self.traverse_directions = [1]*self.dims if traverse_directions is None else traverse_directions
        self.traverse_order = self._get_traverse_order(traverse_order)
        self.endianness = traverse_order if traverse_order in ('little', 'big') else None
        assert self.dims > 0
        assert all(n > 0 for n in self.shape)
        assert all(np.all(np.diff(vals) > 0) for vals in self.values) # values are strictly increasing
        assert len(self.traverse_directions) == len(self.traverse_order) == self.dims
        assert len(self.traverse_order) == len(set(self.traverse_order)) # no duplicates
        assert all([0 <= i < self.dims for i in self.traverse_order])
        assert all([d in (-1, 1) for d in self.traverse_directions])

    def _get_traverse_order(self, order):
        if order is None or order == 'big':
            return list(reversed(range(self.dims)))
        elif order == 'little':
            return list(range(self.dims))
        elif isinstance(order, (np.ndarray, list)):
            return order
        else:
            raise ValueError(order)


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

    def __getitem__(self, index):
        if isinstance(index, (np.ndarray, tuple, list)):
            return self.get_point_at(index)
        elif isinstance(index, int):
            return self.get_nth_point(index)
        else:
            raise TypeError(index)

    def get_nth_index(self, n):
        """ get the nth index in the grid

        The indices can also be used to index into the tensors returned from meshgrid

        Returns:
            an index into the grid, with each element being an integer index
            into the list of possible values along that dimension.
        """
        # traverse the grid with each dimension acting like a digit in a number
        assert 0 <= n < self.num_points
        index = [0] * self.dims
        # traverse from the least significant 'digit' to the most significant
        for d in self.traverse_order:
            n_vals, direction = self.shape[d], self.traverse_directions[d]
            n, i = divmod(n, n_vals)
            if direction == -1:
                i = n_vals - i - 1 # if tracking from the larger end, index into c from that end
            index[d] = i
        assert n == 0
        return tuple(index)

    def get_adjacent_indices(self, index):
        """ get a list of the indices (as tuples) which are adjacent to the given index
        - doesn't include diagonally adjacent indices
        - the indices are presented in an order which obeys traverse_order and traverse_directions
        """
        assert len(index) == self.dims
        # a tuple like 'index' but changed in one dimension
        changed_index = lambda change_d, new_i: tuple(new_i if d == change_d else i for d, i in enumerate(index))
        adj_indices = []
        for d in self.traverse_order:
            n = self.shape[d] # number of values this dimension can take
            i = index[d] # the current index along this dimension
            a = []
            if i > 0: # not on the lower boundary
                a.append(changed_index(d, i-1))
            if i < n-1: # not on the upper boundary
                a.append(changed_index(d, i+1))
            adj_indices += a if self.traverse_directions[d] == 1 else reversed(a)
        return adj_indices

    def get_point_at(self, index):
        """ get the point corresponding to the given grid index

        Args:
            index: an index into the grid, with each element being an integer
                index into the list of possible values along that dimension.
        """
        assert len(index) == self.dims
        return np.array([[self.values[d][index[d]] for d in range(self.dims)]])

    def get_nth_point(self, n):
        """ get the nth point in the grid """
        return self.get_point_at(self.get_nth_index(n))

    def get_indices(self):
        """ get a list of indices into the grid """
        return [self.get_nth_index(n) for n in range(self.num_points)]

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
        return Grid(values, self.traverse_directions, self.traverse_order)


    def subgrid(self, keep_dims):
        """ construct a grid by taking a subset of the dimensions of this grid

        Useful for plotting some dimensions while leaving most at a fixed value

        Args:
            keep_dims: a list of dimension _numbers_ (i.e. indices into
                self.values) to keep.
        """
        assert len(keep_dims) > 0 and all(0 <= d < self.dims for d in keep_dims)
        values = [vals for d, vals in enumerate(self.values) if d in keep_dims]
        traverse_directions = [direction for d, direction in enumerate(self.traverse_directions) if d in keep_dims]
        traverse_order = [keep_dims.index(d) for d in self.traverse_order if d in keep_dims]
        return Grid(values, traverse_directions, traverse_order)

    def meshgrid(self, cartesian_index=True):
        """ return a list of grids like the ones created by `np.meshgrid`

        useful for plotting. May want to first use `subgrid` to extract a 2D
        grid to convert to a meshgrid.

        a meshgrid is list of tensors where each tensor contains values for a
        single dimension. eg in 2D, an n*m grid results in an n*m matrix of x
        values and an n*m matrix of y values.

        Args:
            cartesian_index: the default indexing with `meshgrid` is 'xy' or
                'Cartesian' indexing. This causes indexing like [y,x,z,...] rather
                than 'ij' or 'matrix' indexing which causes indexing like
                [x,y,z,...]. Use Cartesian indexing for plotting.
        """
        if cartesian_index and self.dims > 1:
            # Cartesian indexing, corresponds to meshgrid(..., indexing='xy')
            # basically, the x and y axes are swapped, with the rest remaining the same
            n = np.array(self.shape)
            n[[0,1]] = n[[1,0]] # swap elements 0 and 1
            ids = np.indices(n)
            ids[[0,1]] = ids[[1,0]] # swap elements 0 and 1
        else:
            # matrix indexing, corresponds to meshgrid(..., indexing='ij')
            ids = np.indices(self.shape)

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

    def fun_on_grid(self, f):
        """ return an array of values such that `data[i,j,k,...] = f(get_point_at([i,j,k,...]))`

        The following holds:
        ```
        g = Grid(...)
        grids = g.meshgrid(cartesian_index=False)
        grid.fun_on_grid(f) == points_to_grid(f(grid_to_points(grids)), grids[0].shape)
        ```
        """
        g = Grid(self.values, traverse_directions=[1]*self.dims, traverse_order='big')
        data = f(g.get_points())
        return data.reshape(g.shape)


class RegularGrid(Grid):
    def __init__(self, num_values, bounds, traverse_directions=None, traverse_order='big'):
        """
        Args:
            num_values: a list of the number of possible values in each
                dimension (the grid shape) or a single scalar which applies to
                all dimensions
            bounds: a list of (min, max) for each dimension
            traverse_directions: a list of either -1 or 1 to indicate the
                direction that that dimension should be traversed when iterating
                over the grid. 1 => ascending, -1 => descending.
            traverse_order: see Grid.traverse_order
        """
        shape = [num_values] * len(bounds) if np.isscalar(num_values) else num_values
        assert len(shape) == len(bounds)
        # the possible values along each dimension
        values = [np.linspace(rmin, rmax, num=n) for n, (rmin, rmax) in zip(shape, bounds)]
        super().__init__(values, traverse_directions, traverse_order)





def grid_to_points(grid, endian='big'):
    """ take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix """
    # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
    if endian == 'big':
        return np.vstack([g.T.ravel() for g in grid]).T
    elif endian == 'little':
        return np.vstack([g.ravel() for g in grid]).T
    else:
        raise ValueError(endian)

def vals_to_grid(vals, grid_shape, endian='big'):
    """ take a matrix of values generated with grid_to_points and return it to a grid (tensor) """
    if endian == 'big':
        return vals.reshape(*reversed(grid_shape)).T
    elif endian == 'little':
        return vals.reshape(*grid_shape)
    else:
        raise ValueError(endian)

def fill_meshgrid(grids, f):
    """ fill a tensor with the result of evaluating the function f on the points of the given meshgrid """
    G = np.empty_like(grids[0])
    for index, _ in np.ndenumerate(grids[0]):
        G[index] = f(*[g[index] for g in grids])
    return G




class ElasticNet:
    """
    A class for managing an elastic net which is defined by a multidimensional
    grid of 'control point' locations and a tensor of amplitudes at each control
    point location.
    """
    def __init__(self, grid, elastic_stiffness, range_bounds):
        self.grid = grid
        self.elastic_stiffness = elastic_stiffness
        self.range_bounds = range_bounds
        self._neighbour_kernel = self._construct_neighbour_kernel()

    def random_amplitudes(self):
        """ sample a uniform-random set of amplitudes """
        return np.random.uniform(*self.range_bounds, size=self.grid.shape)

    def elastic_potentials_slow(self, amplitudes, gradient=False):
        """ given a tensor of amplitudes, return a tensor of elastic costs """
        assert amplitudes.shape == self.grid.shape
        if self.elastic_stiffness == 0:
            # special case for speed
            return np.zeros(shape=self.grid.shape, dtype=np.float64)
        else:
            costs = np.empty(shape=self.grid.shape, dtype=np.float64)
            for i in self.grid.get_indices():
                if gradient:
                    costs[i] = self.elastic_potential_gradient_at(i, amplitudes)
                else:
                    costs[i] = self.elastic_potential_at(i, amplitudes)
            return costs

    def elastic_potentials(self, amplitudes):
        r""" calculate the elastic potential energy across the whole net using
        correlation rather than calculating each element individually:

        \begin{align}
        U&=\frac{1}{2}k\sum_i^N (a_{s_i}-a_p)^2\\
        &=\frac{1}{2}k\sum_i^N (a_{s_i}^2+a_p^2-2 a_{s_i}a_p)\\
        &=\frac{1}{2}k\left(\sum_i^N a_{s_i}^2+Na_p^2-2 a_p\sum_i^N a_{s_i}\right)\\
        \end{align}

        """
        k = self._neighbour_kernel.copy()
        k[(1,)*self.grid.dims] = 0 # sum neighbours only
        amplitudes_squared = np.square(amplitudes)
        # reflect needed so that num_neighbours is the same for every element,
        # otherwise too many multiples of p is subtracted at the edges. With
        # reflect on, extra copies of p is added for the edge elements.
        neighbour_sum = scipy.ndimage.correlate(amplitudes, k, mode='reflect')
        neighbour_squared_sum = scipy.ndimage.correlate(amplitudes_squared, k, mode='reflect')
        num_neighbours = 2 * self.grid.dims
        squared_diff = (neighbour_squared_sum + num_neighbours*amplitudes_squared - 2 * amplitudes * neighbour_sum)
        return 0.5 * self.elastic_stiffness * squared_diff

    def elastic_potential_at(self, index, amplitudes):
        r""" calculate the elastic potential energy at a single control point

        doesn't include the 'bending energy' sometimes used in elastic maps
        https://en.wikipedia.org/wiki/Elastic_map

        The following equation is for calculating the magnitude of the force
        exerted on a control point `p` by its neighbours `s_i` with amplitudes
        `a_p` and `a_{s_i}` respectively.
        \begin{equation}
        F\;=\;\sum^N_i F_{\mathrm{spring}_i}
        \;=\; \sum^N_i kd_i
        \;=\; \sum^N_i k\left\|\begin{bmatrix}\mathbf{s}_i\\a_{s_i}\end{bmatrix}-\begin{bmatrix}\mathbf{p}\\a_p\end{bmatrix}\right\|
        \;=\; k\sum^N_i \sqrt{\|\mathbf{s}_i-\mathbf{p}\|^2+(a_{s_i}-a_p)^2}
        \end{equation}

        Because the control points are fixed in place and can only move along
        the 'amplitude' axis, the forces should be projected onto the amplitude
        axis. This simplifies things a lot because only the amplitude axis has
        to be considered, which also makes computation faster since the distance
        to the neighbours is irrelevant.
        $$F=k\sum_i^N|a_{s_i}-a_p|$$

        Integrate to obtain the elastic potential energy:
        $$U=\frac{1}{2}k\sum_i^N (a_{s_i}-a_p)^2$$
        """
        adj = self.grid.get_adjacent_indices(index)
        return 0.5 * self.elastic_stiffness * np.sum(np.square(np.array([amplitudes[a] for a in adj]) - amplitudes[index]))

    def elastic_potential_gradient_at(self, index, amplitudes):
        r"""
        $$\frac{\mathrm d U}{\mathrm d a_p}=-k\sum_i^N(a_{s_i}-a_p)$$
        """
        adj = self.grid.get_adjacent_indices(index)
        return -self.elastic_stiffness * np.sum(np.array([amplitudes[a] for a in adj]) - amplitudes[index])

    def _construct_neighbour_kernel(self):
        """ Construct a kernel which is correlated with the tensor of amplitudes
        to obtain the elastic potential gradient very quickly, rather than
        calculating each element separately.
        """
        d = self.grid.dims
        k = np.zeros(shape=(3,)*d)
        k[(1,)*d] = -2 *d
        for i in range(d):
            idx = [1]*d
            idx[i] = 0
            k[tuple(idx)] = 1
            idx[i] = 2
            k[tuple(idx)] = 1
        return k

    def elastic_potentials_gradient(self, amplitudes):
        """ This is a faster way of calculating a tensor with each cell
        calculated using self.elastic_potential_gradient
        """
        if self.elastic_stiffness == 0:
            # special case for speed
            return np.zeros(shape=self.grid.shape, dtype=np.float64)
        else:
            # at the boundary of the array, the values are reflected so that they
            # are cancelled out, because the term in the sum will be (a_p - a_p)
            return -self.elastic_stiffness * scipy.ndimage.correlate(amplitudes, self._neighbour_kernel, mode='reflect')




#!/usr/bin/env python3

import unittest
import numpy as np
import GPy
import warnings

# local imports
from funbo.grid import *
from funbo import utils

def fun(X):
    """ a function defined in any dimension which returns a very
    different value for every input point to catch cases where
    coordinates are the wrong way round etc.
    """
    return np.array([hash(tuple(x.flatten().tolist())) for x in X]).reshape(-1, 1)


class TestGrid(unittest.TestCase):
    def test_meshgrid(self):

        for e in ('little', 'big'):
            gs = np.meshgrid([1,2], [3,4,5])
            new_gs = [vals_to_grid(grid_to_points(gs, endian=e)[:,i], gs[0].shape, endian=e) for i in range(len(gs))]
            self.assertTrue(np.allclose(gs, new_gs))

        for e in ('little', 'big'):
            gs = np.meshgrid([1,2], [3,4,5], [9, 10, 11, 12])
            new_gs = [vals_to_grid(grid_to_points(gs, endian=e)[:,i], gs[0].shape, endian=e) for i in range(len(gs))]
            self.assertTrue(np.allclose(gs, new_gs))



    def test_grid(self):
        # testing bad data
        self.assertRaises(AssertionError, Grid, [])
        self.assertRaises(AssertionError, Grid, [[1]], [0])
        self.assertRaises(AssertionError, Grid, [[1], [1,2]], [1])
        self.assertRaises(AssertionError, Grid, [[1], [1,2]], [1,1,1])
        self.assertRaises(AssertionError, Grid, [[1]], [])
        self.assertRaises(AssertionError, Grid, [[1]], [[1]], traverse_order=[2,3])
        self.assertRaises(AssertionError, Grid, [[1]], [[1]], traverse_order=[0,0])
        self.assertRaises(AssertionError, Grid, [[1]], [[1]], traverse_order=[1,0,0])
        self.assertRaises(AssertionError, Grid, [[1]], [[1]], [0])
        self.assertRaises(AssertionError, Grid, [[1]], [[1]], [2,2])


        # testing various properties including subgrid and meshgrid
        g = Grid([[1], [1,2]], traverse_order='big')
        ps = np.array([
            [1, 1],
            [1, 2]
        ])
        self.assertTrue(g.num_points == len(ps))
        self.assertTrue(np.allclose(g.get_points(), ps))
        self.assertTrue(np.allclose(np.vstack(p for p in g), ps))
        self.assertTrue(np.allclose(g.subgrid((0,1)).get_points(), g.get_points()))
        self.assertTrue(np.allclose(g.subgrid((0,)).get_points(), np.array([[1]])))
        self.assertTrue(np.allclose(g.subgrid((1,)).get_points(), np.array([[1],[2]])))
        self.assertTrue(np.allclose(np.meshgrid([1], [1,2]), g.meshgrid()))

        g = Grid([[1], [1,2]], [-1, -1], traverse_order='big')
        ps = np.array([
            [1, 2],
            [1, 1]
        ])
        self.assertTrue(g.num_points == len(ps))
        self.assertTrue(np.allclose(g.get_points(), ps))
        self.assertTrue(np.allclose(np.vstack(p for p in g), ps))
        self.assertTrue(np.allclose(g.subgrid((0,1)).get_points(), g.get_points()))
        self.assertTrue(np.allclose(g.subgrid((0,)).get_points(), np.array([[1]])))
        self.assertTrue(np.allclose(g.subgrid((1,)).get_points(), np.array([[2],[1]])))
        self.assertTrue(np.allclose(np.meshgrid([1], [1,2]), g.meshgrid()))

        g = Grid([[1,5], [1,2], [-10, 10]], [1, -1, 1], traverse_order='big')
        ps = np.array([
            [1, 2, -10],
            [1, 2, 10],
            [1, 1, -10],
            [1, 1, 10],
            [5, 2, -10],
            [5, 2, 10],
            [5, 1, -10],
            [5, 1, 10],
        ])
        self.assertTrue(g.num_points == len(ps))
        self.assertTrue(np.allclose(g.get_points(), ps))
        self.assertTrue(np.allclose(np.vstack(p for p in g), ps))
        self.assertTrue(np.allclose(np.meshgrid([1,5], [1,2], [-10,10], indexing='xy'), g.meshgrid()))
        self.assertTrue(np.allclose(np.meshgrid([1,5], [1,2], [-10,10], indexing='ij'), g.meshgrid(cartesian_index=False)))
        self.assertTrue(np.allclose(g.subgrid((0,1)).get_points(), Grid([[1,5], [1,2]], [1,-1], traverse_order='big').get_points()))
        self.assertTrue(np.allclose(g.subgrid((0,2)).get_points(), Grid([[1,5], [-10,10]], [1,1], traverse_order='big').get_points()))
        self.assertTrue(np.allclose(g.subgrid((1,2)).get_points(), Grid([[1,2], [-10,10]], [-1,1], traverse_order='big').get_points()))

        ps = np.array([
            [1,6.5,-10],
            [1,6.5,10],
            [5,6.5,-10],
            [5,6.5,10],
        ])
        self.assertTrue(np.allclose(g.fixed_subgrid([None, 6.5, None]).get_points(), ps))
        ps = np.array([
            [1,6.5,3],
            [5,6.5,3],
        ])
        self.assertTrue(np.allclose(g.fixed_subgrid([None, 6.5, 3]).get_points(), ps))

        g2 = Grid([[1,2], [-10, 10]], [-1, 1], traverse_order='big')
        self.assertTrue(np.allclose(g.subgrid((1,2)).get_points(), g2.get_points()))
        self.assertTrue(np.allclose(np.meshgrid([1,2], [-10,10]), g2.meshgrid()))

        g = Grid([[1,2,9], [-10, 0.5, 10]], [1,1], traverse_order='big')
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='xy'), g.meshgrid()))
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='ij'), g.meshgrid(cartesian_index=False)))

        g = Grid([[1,2,9], [-10, 0.5, 10]], [-1,-1], traverse_order='big')
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='xy'), g.meshgrid()))
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='ij'), g.meshgrid(cartesian_index=False)))

        g = Grid([[1,2,9], [-10, 0.5, 10]], [1,1], traverse_order='little')
        self.assertTrue(np.allclose(g.get_points(), grid_to_points(g.meshgrid(), endian='little')))

        g = Grid([[1,2,9], [-10, 0.5, 10]], [1,1], traverse_order='big')
        self.assertTrue(np.allclose(g.get_points(), grid_to_points(g.meshgrid(), endian='big')))

        # testing fun on grid
        def check_fun_on_grid(data, g, f):
            self.assertEqual(data.shape, g.shape)
            for n in range(g.num_points):
                c = g.get_nth_index(n)
                self.assertEqual(data[c], f(g.get_point_at(c)))

        g = Grid([[1,2,9], [-10, 0.5, 10]])
        check_fun_on_grid(g.fun_on_grid(fun), g, fun)

        g = Grid([[1,2,9], [-10, 0.5, 10]], [-1,1], traverse_order='little')
        check_fun_on_grid(g.fun_on_grid(fun), g, fun)

        g = Grid([[1,2,9], [-10, 0.5, 10]], [-1,1], traverse_order='big')
        check_fun_on_grid(g.fun_on_grid(fun), g, fun)

        grid = g.meshgrid(cartesian_index=False)
        grid_data = vals_to_grid(fun(grid_to_points(grid)), grid[0].shape)
        self.assertTrue(np.allclose(grid_data, g.fun_on_grid(fun)))

        g = Grid([[1,2,9], [-10, 0.5, 10], [7,8]], [-1,1,1], traverse_order='little')
        check_fun_on_grid(g.fun_on_grid(fun), g, fun)

        g = Grid([[1,2,9], [-10, 0.5, 10], [7,8]], [-1,1,1], traverse_order='big')
        check_fun_on_grid(g.fun_on_grid(fun), g, fun)

        grid = g.meshgrid(cartesian_index=False)
        grid_data = vals_to_grid(fun(grid_to_points(grid)), grid[0].shape)
        self.assertTrue(np.allclose(grid_data, g.fun_on_grid(fun)))

        # Accessing with __getitem__
        g = Grid([[1,2,3], [4,5,6], [7,8,9]], traverse_order='big')
        self.assertRaises(TypeError, g.__getitem__, 4.5)
        self.assertTrue(np.array_equal(g[0], g.get_nth_point(0)))
        self.assertTrue(np.array_equal(g[1], g.get_nth_point(1)))
        self.assertTrue(np.array_equal(g[0,0,0], g.get_point_at((0, 0, 0))))
        self.assertTrue(np.array_equal(g[1,0,0], g.get_point_at((1, 0, 0))))
        self.assertTrue(np.array_equal(g[0,0,1], g.get_point_at((0, 0, 1))))
        c = (0,0,1)
        self.assertTrue(np.array_equal(g[0,0,1], g[c]))
        c = [0,0,1]
        self.assertTrue(np.array_equal(g[0,0,1], g[c]))
        c = np.array([0,0,1])
        self.assertTrue(np.array_equal(g[0,0,1], g[c]))


        # getting adjacent indices
        g = Grid([[1,2,3], [4,5,6]], traverse_directions=[1,1], traverse_order='big')
        self.assertTrue(g.get_adjacent_indices((0,0)) == [(0,1), (1,0)])
        self.assertTrue(g.get_adjacent_indices((1,0)) == [(1,1), (0,0), (2,0)])
        self.assertTrue(g.get_adjacent_indices((1,1)) == [(1,0), (1, 2), (0, 1), (2, 1)])

        g = Grid([[1,2,3], [4,5,6]], traverse_directions=[-1,1], traverse_order='big')
        self.assertTrue(g.get_adjacent_indices((1,0)) == [(1,1), (2,0), (0,0)])

        g = Grid([[1,2,3], [4,5,6]], traverse_directions=[1,1], traverse_order='little')
        self.assertTrue(g.get_adjacent_indices((0,0)) == [(1,0), (0,1)])

        g = Grid([[1,2,3], [4,5,6], [7,8,9]], traverse_directions=[1,1,1], traverse_order='big')
        self.assertTrue(g.get_adjacent_indices((1,1,1)) == [(1,1,0), (1,1,2), (1,0,1), (1,2,1), (0,1,1), (2,1,1)])


    def test_elastic_net(self):

        grid = Grid([[1,2,3,4]])
        net = ElasticNet(grid, elastic_stiffness=1, range_bounds=(-10, 10))

        a = np.array([0,0,0,0])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), np.zeros(grid.shape)))
        a = np.array([2,0,1,0])
        c = np.array([2,2.5,1,0.5])
        dc = np.array([2,-3,2,-1])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), dc))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), dc))


        grid = Grid([[1,2,3], [1,2,3]])
        net = ElasticNet(grid, elastic_stiffness=1, range_bounds=(-10, 10))

        a = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0]])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), np.zeros(grid.shape)))
        a = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), np.zeros(grid.shape)))
        a = np.array([[-1,-1,-1],
                      [-1,-1,-1],
                      [-1,-1,-1]])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), np.zeros(grid.shape)))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), np.zeros(grid.shape)))

        a = np.array([[0,0,0],
                      [0,2,0],
                      [0,0,0]])
        # note: when calculating the elastic potential, stiffness is divided by 2
        c  = np.array([[0,2,0],
                       [2,8,2],
                       [0,2,0]])
        dc = np.array([[0,-2 ,0],
                       [-2,8,-2],
                       [0,-2 ,0]])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), dc))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), dc))

        a = np.array([[0,  0, 2],
                      [-2, 4, 0],
                      [0,  0,-2]])
        # note: when calculating the elastic potential, stiffness is divided by 2
        c  = np.array([[2,10,4],
                       [22,42,12],
                       [2,10,4]])
        dc = np.array([[2,-6,4],
                       [-10,18,-4],
                       [2,-2,-4]])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), dc))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), dc))

        grid = Grid([[1,2,3], [1,2,3], [1,2,3]])
        net = ElasticNet(grid, elastic_stiffness=1, range_bounds=(-10, 10))

        a = np.array([
            [[0,  0, 2],
             [-2, 4, 0],
             [0,  0, 0]],
            [[0, 0, 0],
             [0, 3, 2],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 4, 0],
             [0, 1, 0]]
        ])
        # note: when calculating the elastic potential, stiffness is divided by 2
        c = np.array([
            [[2, 10, 6],
             [24, 42.5, 12],
             [2, 8, 0]],
            [[0, 4.5, 4],
             [6.5, 15, 8.5],
             [0, 5, 2]],
            [[0, 8, 0],
             [8, 29, 10],
             [0.5, 6, 0.5]]
        ])
        dc = np.array([
            [[2, -6, 6],
             [-12, 19, -8],
             [2, -4, 0]],
            [[0, -3, -4],
             [-1, 8, 7],
             [0, -4, -2]],
            [[0, -4, 0],
             [-4, 16, -6],
             [-1, 0, -1]]
        ])
        self.assertTrue(np.array_equal(net.elastic_potentials(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_gradient(a), dc))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a), c))
        self.assertTrue(np.array_equal(net.elastic_potentials_slow(a, gradient=True), dc))





if __name__ == '__main__':
    unittest.main()

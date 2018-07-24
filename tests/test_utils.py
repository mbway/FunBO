#!/usr/bin/env python3

import unittest
import numpy as np
import GPy
import warnings

# local imports
from funbo import utils

class TestUtils(unittest.TestCase):
    def test_rbf(self):
        """ test k_RBF against the GPy RBF implementation """

        def k_with_GPy(X, center, sigma, l, return_gradient=False):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*')
                k = GPy.kern.RBF(input_dim=1, variance=sigma**2, lengthscale=l)
                v = k.K(X, center)
                if return_gradient:
                    dv_dr = k.dK_dr_via_X(X, center)
                    # remember in GPy, r is the _scaled_ r
                    # r = |x-center|/l
                    # d/dx |x|=x/|x|
                    diff = X - center
                    dr_dx = diff/(np.abs(diff)*l)
                    dv_dx = dv_dr * dr_dx
                    return v, dv_dx
                else:
                    return v

        sigma, l = 4, 0.5
        center = np.array([[-1.23]])

        # test a single r
        X = np.array([[0.8]])
        self.assertTrue(np.isclose(utils.k_RBF(X, center, sigma, l), k_with_GPy(X, center, sigma, l)))

        # test rs as a column
        X = np.linspace(-4, 2.5, num=10).reshape(-1, 1)
        vals = utils.k_RBF(X, center, sigma, l)
        self.assertTrue(np.allclose(vals, k_with_GPy(X, center, sigma, l)))

        vals, dvals = utils.k_RBF(X, center, sigma, l, True)
        gpyvals, dgpyvals = k_with_GPy(X, center, sigma, l, True)
        self.assertTrue(np.allclose(vals, gpyvals))
        self.assertTrue(np.allclose(dvals, dgpyvals))

    def test_uniform_in_bounds(self):
        np.random.seed(0)

        self.assertRaises(AssertionError, utils.uniform_random_in_bounds, 0, [])
        self.assertRaises(AssertionError, utils.uniform_random_in_bounds, 0, [(0, 1)])

        self.assertRaises(AssertionError, utils.uniform_random_in_bounds, 1, [])

        utils.uniform_random_in_bounds(1, [(0, 1)])
        self.assertRaises(AssertionError, utils.uniform_random_in_bounds, 1, [(1, 0)])
        utils.uniform_random_in_bounds(1, [(1, 1)])

        rs = utils.uniform_random_in_bounds(10_000, [(-2, -1.5), (3, 3.1)])
        rs.shape = (10_000, 2)
        for r in rs:
            self.assertTrue(-2 <= r[0] <= -1.5)
            self.assertTrue(3 <= r[1] <= 3.1)

    def test_grid(self):
        self.assertRaises(AssertionError, utils.Grid, [])
        self.assertRaises(AssertionError, utils.Grid, [[1]], [0])
        self.assertRaises(AssertionError, utils.Grid, [[1], [1,2]], [1])
        self.assertRaises(AssertionError, utils.Grid, [[1], [1,2]], [1,1,1])
        self.assertRaises(AssertionError, utils.Grid, [[1]], [])

        g = utils.Grid([[1], [1,2]])
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

        g = utils.Grid([[1], [1,2]], [-1, -1])
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

        g = utils.Grid([[1,5], [1,2], [-10, 10]], [1, -1, 1])
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
        self.assertTrue(np.allclose(g.subgrid((0,1)).get_points(), utils.Grid([[1,5], [1,2]], [1,-1]).get_points()))
        self.assertTrue(np.allclose(g.subgrid((0,2)).get_points(), utils.Grid([[1,5], [-10,10]], [1,1]).get_points()))
        self.assertTrue(np.allclose(g.subgrid((1,2)).get_points(), utils.Grid([[1,2], [-10,10]], [-1,1]).get_points()))

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

        g2 = utils.Grid([[1,2], [-10, 10]], [-1, 1])
        self.assertTrue(np.allclose(g.subgrid((1,2)).get_points(), g2.get_points()))
        self.assertTrue(np.allclose(np.meshgrid([1,2], [-10,10]), g2.meshgrid()))

        g = utils.Grid([[1,2,9], [-10, 0.5, 10]], [1,1])
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='xy'), g.meshgrid()))
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='ij'), g.meshgrid(cartesian_index=False)))

        g = utils.Grid([[1,2,9], [-10, 0.5, 10]], [-1,-1])
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='xy'), g.meshgrid()))
        self.assertTrue(np.allclose(np.meshgrid([1,2,9], [-10,0.5,10], indexing='ij'), g.meshgrid(cartesian_index=False)))



if __name__ == '__main__':
    unittest.main()

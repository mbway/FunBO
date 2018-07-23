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

if __name__ == '__main__':
    unittest.main()

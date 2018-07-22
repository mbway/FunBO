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

        def k_with_GPy(r, sigma, l):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*')
                k = GPy.kern.RBF(input_dim=1, variance=sigma**2, lengthscale=l)
                v = k.K(np.array([[0]]), np.array([[r]]))
                return np.asscalar(v)

        sigma, l = 4, 0.5

        # test a single r
        r = 0.8
        self.assertTrue(np.isclose(utils.k_RBF(r, sigma, l), k_with_GPy(r, sigma, l)))

        # test rs as a column
        rs = np.linspace(0.4, 2.5, num=10).reshape(-1, 1)
        vals = utils.k_RBF(rs, sigma, l)
        for i, v in enumerate(vals):
            self.assertTrue(np.isclose(v, k_with_GPy(np.asscalar(rs[i]), sigma, l)))

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

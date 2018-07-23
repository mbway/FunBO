#!/usr/bin/env python3

import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# local imports
from funbo import aux_optimisers


def grid_to_points(X, Y):
    ''' take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix '''
    # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
    return np.vstack((X.ravel(), Y.ravel())).T

def points_to_grid(points, grid_size):
    ''' take a matrix of points generated with grid_to_points and return it to a grid'''
    return points.reshape(*grid_size)


class Schwefel:
    """ the negative of the 2-dimensional Schwefel function
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    def __init__(self, dimensions):
        self.bounds = [(-500, 500)] * dimensions
        self.dimensions = dimensions
        self.maximum_x = np.array([420.9687] * self.dimensions)
        self.maximum_y = 0

    def __call__(self, xs):
        assert xs.shape[1] == self.dimensions
        res = 418.9829 * self.dimensions
        parts = []
        for i in range(self.dimensions):
            xi = xs[:,i]
            p = xi*np.sin(np.sqrt(np.abs(xi)))
            parts.append(p.reshape(-1, 1))
        ys = -(res - np.sum(np.hstack(parts), axis=1))
        return ys.reshape(xs.shape[0], 1)

    def plot(self):
        assert self.dimensions == 2
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        nx, ny = 300, 300
        X, Y = np.meshgrid(np.linspace(*self.bounds[0], num=nx), np.linspace(*self.bounds[1], num=ny))
        points = grid_to_points(X, Y)
        Z = self.__call__(points)
        Z = points_to_grid(Z, (nx, ny))
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.show()




class TestAuxOptimisers(unittest.TestCase):
    def test_maximise_random(self):
        np.random.seed(0)

        s = Schwefel(2)
        #s.plot()
        xs, ys = aux_optimisers.maximise_random(s, s.bounds, num_samples=100000, num_take=20)
        # xs and ys are the expected shapes
        self.assertTrue(xs.shape == (20, 2))
        self.assertTrue(ys.shape == (20, 1))
        # ys is sorted in descending order
        self.assertTrue(np.all(ys == np.sort(ys, axis=0)[::-1]))
        # the best is relatively close to the optimum
        best_x, best_y = xs[0], ys[0]
        self.assertTrue(np.isclose(best_y, s.maximum_y, atol=0.1))
        self.assertTrue(np.isclose(np.linalg.norm(best_x - s.maximum_x), 0, atol=1))

    def test_maximise_quasi_Newton(self):
        np.random.seed(0)

        s = Schwefel(2)
        #s.plot()
        xs, ys = aux_optimisers.maximise_quasi_Newton(s, s.bounds, num_its=100, num_take=20, exact_gradient=False, starting_points=None)
        # xs and ys are the expected shapes
        self.assertTrue(xs.shape == (20, 2))
        self.assertTrue(ys.shape == (20, 1))
        # ys is sorted in descending order
        self.assertTrue(np.all(ys == np.sort(ys, axis=0)[::-1]))
        # the best is relatively close to the optimum
        best_x, best_y = xs[0], ys[0]
        self.assertTrue(np.isclose(best_y, s.maximum_y, atol=0.1))
        self.assertTrue(np.isclose(np.linalg.norm(best_x - s.maximum_x), 0, atol=1))

    def test_maximise_random_quasi_Newton(self):
        np.random.seed(0)

        s = Schwefel(2)
        #s.plot()
        x, y = aux_optimisers.maximise_random_quasi_Newton(s, s.bounds, num_random=1000, num_take_random=50, num_bfgs=100, exact_gradient=False)
        # xs and ys are the expected shapes
        self.assertTrue(x.shape == (1, 2))
        self.assertTrue(y.shape == (1, 1))
        # the best is relatively close to the optimum
        self.assertTrue(np.isclose(y, s.maximum_y, atol=0.1))
        self.assertTrue(np.isclose(np.linalg.norm(x - s.maximum_x), 0, atol=1))


if __name__ == '__main__':
    unittest.main()



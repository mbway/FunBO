#!/usr/bin/env python3
"""
Auxiliary optimisers
"""

import numpy as np
import scipy.optimize
import warnings

# local imports
from .utils import uniform_random_in_bounds


def maximise_random_quasi_Newton(f, bounds, num_random, num_take_random, num_bfgs, exact_gradient):
    """ maximise the given function by first sampling randomly, then taking the
        `num_take_random` best samples and using them as starting points in BFGS
        optimisation.

    Args:
        f: a vectorized function defined in `bounds` which can take a matrix of
            values with input points on each row, and return results as rows of
            a (n,1) matrix.
        bounds: a list of [(min, max)] for each dimension of the domain of `f`
        num_random: the number of random samples to evaluate
        num_take_random: the number of best random samples to use as starting
            points for BFGS
        num_bfgs: the number of BFGS iterations to perform
        exact_gradient: whether `f` returns the gradient as well as the values.
            If False then the gradient is approximated numerically.

    Returns:
        `(best_x, best_y)`
    """
    if num_random > 0:
        r_best_xs, r_best_ys = maximise_random(f, bounds, num_random, num_take_random)
    else:
        r_best_xs, r_best_ys = np.empty(shape=(0, len(bounds))), np.empty(shape=(0, 1))

    # can return None, None if every iteration fails
    g_best_x, g_best_y = maximise_quasi_Newton(f, bounds, num_bfgs, 1, exact_gradient, starting_points=r_best_xs)

    if num_random > 0 and (g_best_y is None or r_best_ys[0] > g_best_y):
        return r_best_xs[0], r_best_ys[0]
    else:
        return g_best_x, g_best_y

def maximise_random(f, bounds, num_samples, num_take):
    """ maximise the given function by random sampling and taking the best samples

    Args:
        f: a vectorized function defined in `bounds` which can take a matrix of
            values with input points on each row, and return results as rows of
            a (n,1) matrix.
        bounds: a list of [(min, max)] for each dimension of the domain of `f`
        num_samples: the number of random samples to evaluate
        num_take: the number of best samples to return

    Returns:
        `(best_xs, best_ys)` where `best_xs` and `best_ys` are matrices with `num_take` rows
    """
    assert 0 < num_take <= num_samples
    xs = uniform_random_in_bounds(num_samples, bounds)
    ys = f(xs)
    assert ys.shape == (xs.shape[0], 1)
    best_ids = np.argsort(-ys, axis=0).flatten()  # sorted indices (-ys to sort in descending order)
    take_ids = best_ids[:num_take]
    return xs[take_ids], ys[take_ids]

def maximise_quasi_Newton(f, bounds, num_its, num_take, exact_gradient, starting_points=None):
    """ maximise the given function using L-BFGS-B

    Args:
        f: a vectorized function defined in `bounds` which can take a matrix of
            values with input points on each row, and return results as rows of
            a (n,1) matrix. If `exact_gradient` then `f` should return a tuple
            of (values, gradients).
        bounds: a list of [(min, max)] for each dimension of the domain of `f`
        num_its: the number of iterations to run LBFGS for
        num_take: the number of best samples to return
        exact_gradient: whether `f` returns the gradient as well as the values.
            If False then the gradient is approximated numerically.
        starting_points: provide starting points for the maximisation to start
            at, one per row. None => use `num_its` uniform random starting points

    Returns:
        `(best_xs, best_ys)` where `best_xs` and `best_ys` are matrices with
        `num_take` rows. Or `(None, None)` if all iterations failed.
    """

    assert 0 < num_take <= num_its
    if starting_points is None:
        starting_points = np.empty(shape=(0, len(bounds)))
    assert starting_points.shape[0] <= num_its, 'too many starting points'
    extra_starts = num_its - starting_points.shape[0]
    if extra_starts > 0:
        starting_points = np.vstack((starting_points, uniform_random_in_bounds(extra_starts, bounds)))
    assert starting_points.shape == (num_its, len(bounds))

    xs = []
    neg_ys = []

    # the minimiser passes x as (num_attribs,) but f takes (1,num_attribs)
    opt_fun = lambda x: -f(x.reshape(1, -1))

    for i in range(num_its):
        result = scipy.optimize.minimize(
            fun=opt_fun,
            x0=starting_points[i].reshape(1, -1),
            jac=exact_gradient, # whether f returns the gradient
            bounds=bounds,
            method='L-BFGS-B',  # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
        )
        if not result.success:
            warnings.warn('iteration {}/{} of LBFGSB optimisation failed'.format(i+1, num_its))
        else:
            xs.append(result.x)
            neg_ys.append(result.fun)

    if not xs: # all iterations failed
        return None, None

    # ensure that the inputs lie within the bounds
    # (which may not be the case due to floating point error)
    low_bounds, high_bounds = zip(*bounds)
    xs = [np.clip(x, low_bounds, high_bounds) for x in xs]
    xs = np.vstack(xs)

    neg_ys = np.vstack(neg_ys)
    ys = -neg_ys

    best_ids = np.argsort(neg_ys, axis=0).flatten()  # sorted indices (neg_ys is negative already so sort in descending order)
    take_ids = best_ids[:num_take]

    return xs[take_ids], ys[take_ids]


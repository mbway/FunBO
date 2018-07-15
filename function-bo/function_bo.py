#!/usr/bin/env python3

import numpy as np
import scipy
import GPy
from collections import namedtuple

def interpolated(xs, ys):
    return scipy.interpolate.interp1d(xs, ys, kind='linear')


class Optimiser:
    def __init__(self, objective, bounds, pre_phase_trials, desired_extremum='min'):
        """
        Args:
            objective: a function which takes a function to test and returns a
            list of local rewards and a single global reward: [(x, R_l(x))], R_g
        """
        self.objective = objective
        self.bounds = bounds
        self.pre_phase_trials = pre_phase_trials
        assert desired_extremum in ('min', 'max')
        self.desired_extremum = desired_extremum


        # runtime data
        self.has_run = False
        self.trials = []

    Trial = namedtuple('Trial', ['f', 'R_ls', 'R_g', 'surrogate'])


    def is_maximising(self):
        return self.desired_extremum == 'max'
    def is_minimising(self):
        return self.desired_extremum == 'min'
    def _is_better(self, a, b):
        """ return whether a is better than b (depending on the desired extremum) """
        if self.is_maximising():
            return a > b
        else:
            return a < b


    def select_random(self, kernel=None, mu=None):
        """ Sample a function from a GP prior using the given mean function and kernel

        Args:
            kernel: the GPy kernel to use to generate the prior
            mu: the mean function to use. None=> zero function

        Returns:
            a function which can be evaluated anywhere in the domain
        """
        _, xmin, xmax = self.bounds
        num = 100
        xs = np.linspace(xmin, xmax, num=num).reshape(-1, 1) # points to sample at

        # mean function
        if mu is None:
            mus = np.zeros(num)
        else:
            mus = np.array([mu(np.asscalar(x)) for x in xs])

        # covariance matrix
        if kernel is None:
            kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
        C = kernel.K(xs, xs)

        ys = np.random.multivariate_normal(mean=mus, cov=C, size=1)
        return interpolated(xs.flatten(), ys.flatten())

    def select_bayes(self):
        X = [] # (x, y)
        R = [] # R_l(x,y)
        for t in self.trials:
            X += [(x, t.f(x)) for x, r in t.R_ls]
            R += [r for x, r in t.R_ls]
        X = np.array(X)
        R = np.array(R).reshape(-1,1)

        kernel = GPy.kern.RBF(input_dim=2, ARD=True, variance=1.0, lengthscale=1.0)
        surrogate = GPy.models.GPRegression(X, R, kernel)
        surrogate.optimize_restarts(num_restarts=10)

        f = lambda x: 0

        return f, surrogate



    def get_incumbent(self):
        assert self.trials, 'no trials'
        inc_i, inc = 0, self.trials[0]
        for i, t in enumerate(self.trials[1:]):
            if self._is_better(t.R_g, inc.R_g):
                inc_i, inc = i, t
        return inc_i, inc



    def run(self, max_trials):
        assert not self.has_run, 'optimiser already run'
        self.has_run = True

        for i in range(max_trials):
            if i < self.pre_phase_trials:
                f = self.select_random()
                surrogate = None
            else:
                f, surrogate = self.select_bayes()

            R_ls, R_g = self.objective(f)
            t = Optimiser.Trial(f, R_ls, R_g, surrogate)
            self.trials.append(t)


#!/usr/bin/env python3

import time
import numpy as np
import scipy
import GPy
from collections import namedtuple
import warnings

import turbo as tb
import turbo.modules as tm

class PiecewiseFunction:
    def __init__(self, xs, ys, interpolation='linear'):
        self.control_xs = xs
        self.control_ys = ys
        self.f = scipy.interpolate.interp1d(xs, ys, kind=interpolation)

    def __str__(self):
        cs = ', '.join(['({:.2f}, {:.2f})'.format(x, y) for x, y in zip(self.control_xs, self.control_ys)])
        return 'PiecewiseFunction<{}>'.format(cs)

    def __call__(self, x):
        return self.f(x)


def maximise(f, range_):
    """ maximise the 1D real-valued function f over the given range

    Args:
        range_: (min, max)
    """
    bounds = tb.Bounds([('_', range_[0], range_[1])])
    aux_optimiser = tm.RandomAndQuasiNewton(num_random=1000, grad_restarts=8, start_from_best=4)
    best_y, maximisation_info = aux_optimiser(bounds, f)
    return np.asscalar(best_y), maximisation_info


def k_RBF(r, sigma, l):
    return sigma**2 * np.exp(-0.5 * np.square(r/l))


def UCB(X, beta, surrogate, maximising):
    """
    Args:
        X: the inputs for the surrogate to predict at (one per row)
        beta: the exploration-exploitation trade-off parameter
        surrogate: the surrogate model to use for predictions
        maximising: whether the optimiser is maximising or minimising (minimising => -LCB rather than UCB)
    """
    mu, var = surrogate.predict(X)
    mu = mu.flatten()
    sigma = np.sqrt(np.clip(var, 0, np.inf)).flatten()
    sf = 1 if maximising else -1 # scale factor
    # in this form it is clearer that the value is the negative LCB when minimising
    # sf * (mus + sf * beta * sigmas)
    return sf * mu + beta * sigma

def tracking_weights(X, prev_xy, l):
    """ for the input points X, calculate the tracking weight using an RBF function centered at prev_xy
    Args:
        prev_xy: an array with a single row corresponding to (x,y) of the
            previous chosen function value so that this function can stay close
            to it.
    """
    assert prev_xy.shape == (1, X.shape[1])
    rs = X - prev_xy # subtract from every row
    rs = np.linalg.norm(rs, axis=1)
    return k_RBF(rs.reshape(-1, 1), sigma=1, l=l).flatten()


class GPPriorSelectConfig:
    """
    Attributes:
        mu: the mean function for the Gaussian process prior which is
            sampled from. None => use the zero function
        kernel: the kernel for the Gaussian process prior which is sampled
            from. The type of kernel and its parameters greatly affect the
            behaviour of the sample function.
    """
    def __init__(self, domain_bounds):
        self.mu = None
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0) # TODO: not multi-dimensional
        _, xmin, xmax = domain_bounds # TODO: not multi dimensional
        self.control_xs = np.linspace(xmin, xmax, num=100)

class RandomSelectConfig:
    """
    Attributes:
    """
    def __init__(self, domain_bounds):
        _, xmin, xmax = domain_bounds # TODO: not multi dimensional
        self.control_xs = np.linspace(xmin, xmax, num=10)

class BayesSelectConfig:
    r"""
    Attributes:
        reward_combo: the weighting $(\alpha, \beta)$ (both $\in[0,1]$) for the
            local rewards $R_l$ and global reward $R_g$ in the convex combination
            $\alpha*R_l + \beta*R_g$
        kernel: the kernel to use for the surrogate model
        surrogate_optimise_iterations: the number of optimiser iterations to fit the surrogate model to the data
        control_xs: the x values to calculate the optimal y value and use (x,y) as a control point for the function sample
        acquisition: the acquisition function to use
        acquisition_param: the exploration/exploitation trade-off parameter passed to the acquisition function
        tracking_l: the length scale to use in the kernel used in calculating the tracking weights
    """
    def __init__(self, domain_bounds):
        self.reward_combo = (1, 0)
        self.kernel = GPy.kern.RBF(input_dim=2, ARD=True) # TODO: not multi-dimensional
        self.surrogate_optimise_iterations = 10
        _, xmin, xmax = domain_bounds # TODO: not multi dimensional
        self.control_xs = np.linspace(xmin, xmax, num=50)
        self.acquisition = UCB
        self.acquisition_param = 1.0
        self.tracking_l = 10 #TODO: might want another kernel other than RBF


class CoordinatorBase:
    """ Describes a Bayesian optimisation strategy with the implementation
        details abstracted away.

    By inheriting from this class and overriding get_config, the behaviour of
    the optimiser can be completely re-configured.
    """
    def get_max_trials(self):
        """ return the maximum number of trials (can return None if unknown) """
        raise NotImplemented()

    def get_config(self, trial_num):
        """ get the selection configuration for this trial

        Returns:
            None => finish the optimisation
            RandomSelectConfig => perform a random selection with the given configuration
            GPPriorSelectConfig => perform a random selection with the given configuration
            BayesSelectConfig => perform a Bayesian selection with the given configuration
        """
        raise NotImplemented()


class Coordinator(CoordinatorBase):
    """ A coordinator which follows the standard pattern of some number of
    pre_phase trials, then switching to Bayesian optimisation trials for a set
    number of trials.
    """
    def __init__(self, domain_bounds, pre_phase_trials, max_trials):
        self.domain_bounds = domain_bounds
        self.pre_phase_trials = pre_phase_trials
        self.max_trials = max_trials

    def get_max_trials(self):
        """ return the maximum number of trials (can return None if unknown) """
        return self.max_trials

    def get_config(self, trial_num):
        """ get the selection configuration for this trial

        Returns:
            None => finish the optimisation
            RandomSelectConfig => perform a random selection with the given configuration
            GPPriorSelectConfig => perform a random selection with the given configuration
            BayesSelectConfig => perform a Bayesian selection with the given configuration
        """
        if trial_num < self.pre_phase_trials:
            return self.get_pre_phase_config(trial_num)
        elif trial_num < self.max_trials:
            return self.get_bayes_config(trial_num)
        else:
            return None

    def get_pre_phase_config(self, trial_num):
        """ override this method to change the behaviour """
        return RandomSelectConfig()

    def get_bayes_config(self, trial_num):
        """ override this method to change the behaviour """
        return BayesSelectConfig(self.domain_bounds)



class Optimiser:
    """
    Attributes:
        clip_range: whether to bound the outputs of any generated function to lie strictly within range_bounds
    """
    def __init__(self, objective, domain_bounds, range_bounds, coordinator, desired_extremum='min'):
        """
        Args:
            objective: a function which takes a function to test and returns a
                list of local rewards and a single global reward: [(x, R_l(x))], R_g
            domain_bounds: the bounds for the inputs to the function to optimise [(name, min, max)]
            range_bounds: the bounds of the function output (min, max)
            coordinator: see `Coordinator`
        """
        self.objective = objective
        self.domain_bounds = domain_bounds # TODO: wrap in turbo bounds
        self.range_bounds = range_bounds
        assert len(range_bounds) == 2
        self.coordinator = coordinator
        assert desired_extremum in ('min', 'max')
        self.desired_extremum = desired_extremum

        # settings
        self.clip_range = True

        # runtime data
        self.has_run = False
        self.trials = []

    Trial = namedtuple('Trial', ['trial_num', 'config', 'f', 'R_ls', 'R_g', 'surrogate'])


    def is_maximising(self):
        return self.desired_extremum == 'max'
    def is_minimising(self):
        return self.desired_extremum == 'min'
    def _is_better(self, a, b):
        """ return whether a is better than b (depending on the desired extremum) """
        return bool(a > b) if self.is_maximising() else bool(a < b)

    def get_incumbent(self):
        assert self.trials, 'no trials'
        inc_i, inc = 0, self.trials[0]
        for i, t in enumerate(self.trials[1:]):
            if self._is_better(t.R_g, inc.R_g):
                inc_i, inc = i, t
        return inc_i, inc


    def select_random(self, c):
        """ Sample a function using random control points for a piecewise function

        The difference between this function and select_GP_prior is that each
        control point here is independent.
        """
        ys = [np.random.uniform(*self.range_bounds) for _ in c.control_xs]
        return PiecewiseFunction(c.control_xs, ys, interpolation='quadratic')

    def select_GP_prior(self, c):
        """ Sample a function from a GP prior using the given mean function and kernel

        Args:
            c (RandomSelectConfig): the configuration to use for this selection

        Returns:
            a function which can be evaluated anywhere in the domain
        """
        xs = c.control_xs.reshape(-1, 1) # points to sample at
        # mean function
        mus = np.zeros(len(xs)) if c.mu is None else np.array([c.mu(np.asscalar(x)) for x in xs])
        # covariance matrix
        C = c.kernel.K(xs, xs)
        ys = np.random.multivariate_normal(mean=mus, cov=C, size=1)
        if self.clip_range:
            ys = np.clip(ys, *self.range_bounds)
        return PiecewiseFunction(xs.flatten(), ys.flatten())

    def extract_function(self, surrogate, c):
        ys = []
        prev_xy = None
        for x in c.control_xs:
            def acq(ys):
                xs = np.repeat(x.reshape(1, -1), ys.shape[0], axis=0) # stack copies of x as rows
                X = np.hstack((xs, ys))
                As = c.acquisition(X, c.acquisition_param, surrogate=surrogate, maximising=self.is_maximising())
                As -= np.min(As) # shift to make 0 the worst possible value, so that at w=0, A*w is the worst possible value.
                if prev_xy is None:
                    return As
                else:
                    ws = tracking_weights(X, prev_xy, l=c.tracking_l)
                    return As * ws
            best_y, _ = maximise(acq, self.range_bounds)
            ys.append(best_y)
            prev_xy = np.append(x, best_y).reshape(1, -1)

        if self.clip_range:
            ys = np.clip(np.array(ys), *self.range_bounds)
        return PiecewiseFunction(c.control_xs, ys)

    def select_bayes(self, c):
        """
        Args:
            c (BayesSelectConfig): the configuration for the selection
        """
        X = [] # (x, y)
        R = [] # R_l(x,y)
        # weighting for the convex combination of the local and global reward to model
        alpha, beta = c.reward_combo
        assert (0 <= alpha <= 1) and (0 <= beta <= 1) and (np.isclose(alpha+beta, 1))

        for t in self.trials:
            X += [(x, t.f(x)) for x, r in t.R_ls]
            R += [alpha*r + beta*t.R_g for x, r in t.R_ls]
        X = np.array(X)
        R = np.array(R).reshape(-1,1)

        surrogate = GPy.models.GPRegression(X, R, c.kernel)
        with warnings.catch_warnings(record=True) as ws:
            # num_restarts is actually the number of iterations
            surrogate.optimize_restarts(num_restarts=c.surrogate_optimise_iterations, parallel=True, verbose=False)

        f = self.extract_function(surrogate, c)
        return f, surrogate


    def run(self, quiet=False):
        assert not self.has_run, 'optimiser already run'
        self.has_run = True

        start = time.time()
        trial_num = 0
        while True:
            if not quiet:
                print('trial {}/{}'.format(trial_num+1, self.coordinator.get_max_trials() or '?'))
            c = self.coordinator.get_config(trial_num)

            if c is None:
                break # finished
            elif isinstance(c, RandomSelectConfig):
                f = self.select_random(c)
                surrogate = None
            elif isinstance(c, GPPriorSelectConfig):
                f = self.select_GP_prior(c)
                surrogate = None
            elif isinstance(c, BayesSelectConfig):
                f, surrogate = self.select_bayes(c)
            else:
                raise ValueError()

            R_ls, R_g = self.objective(f)
            t = Optimiser.Trial(trial_num, c, f, R_ls, R_g, surrogate)
            self.trials.append(t)
            trial_num += 1

        print('optimisation finished: {} trials in {:.1f} seconds'.format(trial_num, time.time()-start))


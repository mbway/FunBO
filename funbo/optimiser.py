#!/usr/bin/env python3

import time
import numpy as np
import scipy
import GPy
from collections import namedtuple
import warnings

import turbo as tb
import turbo.modules as tm

#TODO: hyperparameter continuity





class Optimiser(FixedAttributes):
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
        assert domain_bounds[2] > domain_bounds[1]

        self.range_bounds = range_bounds
        assert len(range_bounds) == 2
        assert range_bounds[1] > range_bounds[0]

        self.coordinator = coordinator

        assert desired_extremum in ('min', 'max')
        self.desired_extremum = desired_extremum

        # settings
        self.clip_range = True

        # runtime data
        self.has_run = False
        self.trials = []

        # prevent creating new attributes to avoid typos
        self.fixed_attributes = True

    Trial = namedtuple('Trial', ['trial_num', 'config', 'f', 'R_ls', 'R_g', 'surrogate', 'eval_info'])
    Trial.__doc__ = """ Holds data for a single trial/iteration of the optimiser

    Attributes:
        eval_info: optional data returned from the objective function
    """


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
        assert isinstance(c, RandomSelectConfig)
        ys = [np.random.uniform(*self.range_bounds) for _ in c.control_xs]
        return PiecewiseFunction(c.control_xs, ys,
                                 interpolation=c.interpolation,
                                 clip_range=self.range_bounds if self.clip_range else None)

    def select_GP_prior(self, c):
        """ Sample a function from a GP prior using the given mean function and kernel

        Args:
            c (RandomSelectConfig): the configuration to use for this selection

        Returns:
            a function which can be evaluated anywhere in the domain
        """
        assert isinstance(c, GPPriorSelectConfig)
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
        assert isinstance(c, BayesSelectConfig)
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


        # don't initialise the model until the initial hyperparameters have been set
        # will always raise RuntimeWarning("Don't forget to initialize by self.initialize_parameter()!")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*initialize_parameter.*')
            surrogate = c.surrogate_class(X, R, initialize=False, **c.surrogate_model_params)

        # these steps for initialising a model from stored parameters are from https://github.com/SheffieldML/GPy
        surrogate.update_model(False)  # prevents the GP from fitting to the data until we are ready to enable it manually
        surrogate.initialize_parameter()  # initialises the hyperparameter objects
        if c.surrogate_init_params is not None:
            surrogate[:] = c.surrogate_init_params
        surrogate.update_model(True)

        # the current parameters are used as one of the starting locations (as of the time of writing)
        # https://github.com/sods/paramz/blob/master/paramz/model.py
        with warnings.catch_warnings(record=True) as ws:
            # num_restarts is actually the number of iterations
            r = c.surrogate_optimise_params.get('num_restarts', None)
            if r is None or r > 0:
                surrogate.optimize_restarts(**c.surrogate_optimise_params)

        if c.surrogate_optimise_params.get('verbose'):
            for w in ws:
                print(w)

        f = self.extract_function(surrogate, c)
        return f, surrogate

    def select(self, c):
        if isinstance(c, RandomSelectConfig):
            f = self.select_random(c)
            surrogate = None
        elif isinstance(c, GPPriorSelectConfig):
            f = self.select_GP_prior(c)
            surrogate = None
        elif isinstance(c, BayesSelectConfig):
            f, surrogate = self.select_bayes(c)
        else:
            raise ValueError()
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

            f, surrogate = self.select(c)

            res = self.objective(f)
            if len(res) == 2:
                R_ls, R_g = res
                eval_info = None
            elif len(res) == 3:
                R_ls, R_g, eval_info = res
            else:
                raise ValueError('wrong number of values ({}) returned from objective function'.format(len(res)))

            t = Optimiser.Trial(trial_num, c, f, R_ls, R_g, surrogate, eval_info)
            self.trials.append(t)
            self.coordinator.trial_finished(trial_num, t)
            trial_num += 1

        print('optimisation finished: {} trials in {:.1f} seconds'.format(trial_num, time.time()-start))


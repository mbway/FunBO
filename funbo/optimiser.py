#!/usr/bin/env python3

import time
import numpy as np
import scipy
import GPy
from collections import namedtuple
import warnings

# local imports
from .utils import FixedAttributes, PiecewiseFunction, k_RBF, show_warnings
from .acquisition import RBF_weighted
from .coordinator import *


class Trial(FixedAttributes):
    """ Holds data for a single trial/iteration of the optimiser

    Attributes:
        user_info: optional data returned from the objective function
    """
    slots = ('trial_num', 'config', 'f', 'R_ls', 'R_g', 'surrogate', 'user_info', 'timing_info')

    class TimingInfo(FixedAttributes):
        """
        realistically, total should equal selection + evaluation and selection
        should equal fitting + extraction, but there may by some extra
        computation which takes some time.

        fitting and extraction may be None if the trial was not Bayes selected
        """
        slots = ('total', 'selection', 'fitting', 'extraction', 'evaluation')
        def __init__(self):
            self._null_init() # set everything to None



class Optimiser(FixedAttributes):
    """ Continuous Function Bayesian Optimisation """
    slots = (
        'objective', 'domain_names', 'domain_bounds', 'range_bounds',
        'coordinator', 'desired_extremum', 'clip_range', 'has_run', 'trials'
    )

    def __init__(self, objective, domain_bounds, range_bounds, desired_extremum='min', clip_range=True):
        """
        Args:
            objective: a function which takes a function to test and returns a
                list of local rewards and a single global reward: [(x, R_l(x))], R_g
            domain_bounds: the bounds for the inputs to the function to optimise [(name, min, max)]
            coordinator: see `Coordinator`
            range_bounds: the bounds of the function output (min, max)
            clip_range: whether to ensure that functions generated by the
                optimiser lie strictly within `range_bounds`. This is
                achieved by clipping the output of the functions.
        """
        self.objective = objective

        # separate the names and bounds because usually only the bounds are required
        self.domain_names = [b[0] for b in domain_bounds]
        self.domain_bounds = [(b[1], b[2]) for b in domain_bounds] # TODO: wrap in turbo bounds
        assert all(b[0] < b[1] for b in self.domain_bounds)

        #TODO: maybe support optional range_name by passing (name, ymin, ymax) as range_bounds
        self.range_bounds = range_bounds
        assert len(range_bounds) == 2
        assert range_bounds[0] < range_bounds[1]

        assert desired_extremum in ('min', 'max')
        self.desired_extremum = desired_extremum
        self.clip_range = clip_range

        # runtime data
        self.has_run = False
        self.trials = []


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


    def select_random_CP(self, c):
        """ Sample a function using random control points for a piecewise function

        The difference between this function and select_GP_prior is that each
        control point here is independent.
        """
        assert isinstance(c, RandomCPSelectConfig)
        ys = [np.random.uniform(*self.range_bounds) for _ in c.control_xs]
        return PiecewiseFunction(c.control_xs, ys, interpolation=c.interpolation,
                                 clip_range=self.range_bounds if self.clip_range else None)

    def select_GP_prior(self, c):
        """ Sample a function from a GP prior using the given mean function and kernel

        Args:
            c (GPPriorSelectConfig): the configuration to use for this selection

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
        return PiecewiseFunction(xs.flatten(), ys.flatten(), interpolation=c.interpolation,
                                 clip_range=self.range_bounds if self.clip_range else None)


    def weighted_function_extraction(self, surrogate, c):
        assert isinstance(c, WeightedExtractionConfig)
        ys = []
        prev_xy = None
        for x in c.control_xs:

            def get_X(ys):
                xs = np.repeat(x.reshape(1, -1), ys.shape[0], axis=0) # stack copies of x as rows
                X = np.hstack((xs, ys))
                return X

            X = get_X(np.linspace(*self.range_bounds, num=500).reshape(-1, 1))
            worst_acq = np.min(c.acquisition(X, surrogate=surrogate,
                                             maximising=self.is_maximising(),
                                             **c.acquisition_params))

            def acq(ys, return_gradient=False):
                """ with x fixed, given ys as rows calculate the weighted
                acquisition function for the pairs of x and y
                """
                X = get_X(ys)
                res = c.acquisition(X, surrogate=surrogate,
                                   maximising=self.is_maximising(),
                                   return_gradient=return_gradient,
                                   **c.acquisition_params)
                acq, dacq_dx = res if return_gradient else (res, None)
                # shift to make 0 the worst possible value, so that at w=0,
                # A*w is the worst possible value.
                # can't subtract min(acq) because ys may not be many points over the whole domain
                acq -= worst_acq

                if prev_xy is not None: # can only weight after the first sample
                    res = RBF_weighted(X, acq, dacq_dx, center=prev_xy, sigma=1, l=c.tracking_l)
                    acq, dacq_dx = res if return_gradient else (res, None)

                if return_gradient:
                    dacq_dy = dacq_dx[0, -1:] # take only the derivative w.r.t y as a single-element array
                    return acq, dacq_dy
                else:
                    return acq

            best_y, best_acq = c.aux_optimiser(acq, [self.range_bounds], **c.aux_optimiser_params)
            ys.append(np.asscalar(best_y))
            prev_xy = np.append(x, best_y).reshape(1, -1)

        return PiecewiseFunction(c.control_xs, ys, interpolation=c.interpolation,
                                 clip_range=self.range_bounds if self.clip_range else None)

    def fit_surrogate(self, data, c):
        """ fit a surrogate model to the given data

        Args:
            data: (input, output) for the surrogate to fit to
            c (SurrogateConfig): the configuration for the surrogate
        """
        assert isinstance(c, SurrogateConfig)

        # don't initialise the model until the initial hyperparameters have been set
        # will always raise RuntimeWarning("Don't forget to initialize by self.initialize_parameter()!")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*initialize_parameter.*')
            surrogate = c.model_class(data[0], data[1], initialize=False, **c.init_params)

        # these steps for initialising a model from stored parameters are from https://github.com/SheffieldML/GPy
        surrogate.update_model(False)  # prevents the GP from fitting to the data until we are ready to enable it manually
        surrogate.initialize_parameter()  # initialises the hyperparameter objects

        if c.initial_hyper_params is None:
            surrogate.randomize()

        elif c.initial_hyper_params == 'last':
            # hyperparameter continuity, use the previous model's hyperparameters as a starting point
            last_params = None
            for t in reversed(self.trials):
                if t.surrogate is not None:
                    last_params = t.surrogate[:]
                    break

            if last_params is None:
                surrogate.randomize()
            else:
                surrogate[:] = last_params
        else:
            surrogate[:] = c.initial_hyper_params

        surrogate.update_model(True)

        # the current parameters are used as one of the starting locations (as of the time of writing)
        # https://github.com/sods/paramz/blob/master/paramz/model.py
        with warnings.catch_warnings(record=True) as ws:
            # num_restarts is actually the number of iterations
            r = c.optimise_params.get('num_restarts', None)
            if r is None or r > 0:
                surrogate.optimize_restarts(**c.optimise_params)

        if c.optimise_params.get('verbose'):
            show_warnings(ws)

        return surrogate

    def select_bayes(self, c, timings):
        """
        Args:
            c (BayesSelectConfig): the configuration for the selection
            timings (Trial.TimingInfo): an object to fill out the relevant timing info
        """
        assert isinstance(c, BayesSelectConfig)

        X = [] # (x, y)
        R_l = [] # R_l(x,y)
        for t in self.trials:
            X += [(x, t.f(x)) for x, r in t.R_ls] # TODO: probably won't work with multiple dimensions in x
            R_l += [r for x, r in t.R_ls]
        X = np.array(X)
        R_l = np.array(R_l).reshape(-1, 1)

        timer = time.perf_counter()
        surrogate = self.fit_surrogate((X, R_l), c.surrogate_config)
        timings.fitting = time.perf_counter()-timer

        timer = time.perf_counter()
        f = self.extract_function(surrogate, c.extraction_config)
        timings.extraction = time.perf_counter()-timer

        return f, surrogate

    def extract_function(self, surrogate, c):
        """ extract a function from the given surrogate using the given configuration
        """
        if isinstance(c, WeightedExtractionConfig):
            return self.weighted_function_extraction(surrogate, c)
        else:
            raise ValueError(c)

    def select(self, c, timings):
        """ Select a function using the given configuration
        Args:
            c: the configuration to use for selection
            timings (Trial.TimingInfo): an object to fill out the relevant timing info
        """
        timer = time.perf_counter()
        if isinstance(c, RandomCPSelectConfig):
            f = self.select_random_CP(c)
            surrogate = None
        elif isinstance(c, GPPriorSelectConfig):
            f = self.select_GP_prior(c)
            surrogate = None
        elif isinstance(c, BayesSelectConfig):
            f, surrogate = self.select_bayes(c, timings)
        else:
            raise ValueError(c)

        timings.selection = time.perf_counter() - timer
        return f, surrogate


    def run(self, coordinator, quiet=False):
        assert not self.has_run, 'optimiser already run'
        self.has_run = True

        start = time.time()
        trial_num = 0
        while True:
            if not quiet:
                print('trial {}/{}'.format(trial_num+1, coordinator.get_max_trials() or '?'))

            timings = Trial.TimingInfo()
            trial_start = time.perf_counter()

            c = coordinator.get_config(trial_num)
            if c is None:
                break # finished

            f, surrogate = self.select(c, timings)

            eval_start = time.perf_counter()
            res = self.objective(f)
            timings.evaluation = time.perf_counter() - eval_start

            if len(res) == 2:
                R_ls, R_g = res
                user_info = None
            elif len(res) == 3:
                R_ls, R_g, user_info = res
            else:
                raise ValueError('wrong number of values ({}) returned from objective function'.format(len(res)))

            timings.total = time.perf_counter()-trial_start
            t = Trial(trial_num, c, f, R_ls, R_g, surrogate, user_info, timings)
            self.trials.append(t)
            trial_num += 1

        print('optimisation finished: {} trials in {:.1f} seconds'.format(trial_num, time.time()-start))


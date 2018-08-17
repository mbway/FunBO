#!/usr/bin/env python3

import time
import numpy as np
import scipy
import GPy
from collections import namedtuple

# local imports
from .utils import FixedAttributes, InterpolatedFunction, Timer, uniform_random_in_bounds
from .acquisition import RBF_weighted
from .coordinator import *


class Trial(FixedAttributes):
    """ Holds data for a single trial/iteration of the optimiser

    Attributes:
    """
    slots = ('trial_num', 'config', 'f', 'R_ls', 'R_g', 'surrogate', 'duration', 'selection', 'evaluation')

    def is_bayes(self):
        return type(self.config) == BayesSelectConfig

    class SelectionInfo(FixedAttributes):
        """
        Attributes:
            duration: the time taken to select the function
            fitting_info: a FittingInfo instance. None if randomly selected
            extraction_info: an ExtractionInfo instance. None if randomly selected
        """
        slots = ('duration', 'fitting', 'extraction')
        def __init__(self, duration, fitting_info, extraction_info):
            self.duration = duration
            self.fitting = fitting_info
            self.extraction = extraction_info

    class FittingInfo(FixedAttributes):
        """
        Attributes:
            duration: the time taken to fit the surrogate model
            data_set_size: the number of samples the surrogate used as training data
        """
        slots = ('duration', 'data_set_size')
        def __init__(self, duration, data_set_size):
            self.duration = duration
            self.data_set_size = data_set_size

    class ExtractionInfo(FixedAttributes):
        """
        Attributes:
            duration: the time taken to extract the function
            acq_evals: the number of points the acquisition function was queried
                at during the extraction
            acq_gradient_evals: the number of points the acquisition gradient
                was analytically computed at during the extraction
            acq_total_time: the total amount of time spent querying the
                acquisition function during the extraction
        """
        slots = ('duration', 'acq_evals', 'acq_gradient_evals', 'acq_total_time')
        def __init__(self):
            self.duration = 0
            self.acq_evals = 0
            self.acq_gradient_evals = 0
            self.acq_total_time = 0

        def get_overhead_duration(self):
            return self.duration - self.acq_total_time

    class EvaluationInfo(FixedAttributes):
        """
        Attributes:
            duration: the time taken to evaluate the function
            user_info: optional data returned from the objective function
        """
        slots = ('duration', 'user_info')
        def __init__(self, duration, user_info):
            self.duration = duration
            self.user_info = user_info




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
        self.domain_bounds = [(b[1], b[2]) for b in domain_bounds]
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


    ################################
    # Random Selection
    ################################

    def select_random_CP(self, c):
        """ Sample a function using random control points for a piecewise function

        The difference between this function and select_GP_prior is that each
        control point here is independent.
        """
        assert type(c) is RandomCPSelectConfig
        t = Timer()
        ys = np.array([np.random.uniform(*self.range_bounds) for _ in range(len(c.control_xs))]).reshape(-1, 1)
        f = InterpolatedFunction(c.control_xs, ys, interpolation=c.interpolation,
                                    clip_range=self.range_bounds if self.clip_range else None)
        selection_info = Trial.SelectionInfo(t.stop(), fitting_info=None, extraction_info=None)
        return (f, selection_info)

    def select_GP_prior(self, c):
        """ Sample a function from a GP prior using the given mean function and kernel

        Args:
            c (GPPriorSelectConfig): the configuration to use for this selection

        Returns:
            a function which can be evaluated anywhere in the domain
        """
        assert type(c) is GPPriorSelectConfig
        t = Timer()
        xs = c.control_xs.get_points()
        # mean function
        mus = np.zeros(len(xs)) if c.mu is None else np.array([c.mu(np.asscalar(x)) for x in xs])
        # covariance matrix
        C = c.kernel.K(xs, xs)
        ys = np.random.multivariate_normal(mean=mus, cov=C, size=1).reshape(-1, 1)
        f = InterpolatedFunction(c.control_xs, ys, interpolation=c.interpolation,
                                    clip_range=self.range_bounds if self.clip_range else None)
        selection_info = Trial.SelectionInfo(t.stop(), fitting_info=None, extraction_info=None)
        return (f, selection_info)


    ################################
    # Function Extraction
    ################################

    def extract_function(self, surrogate, c):
        """ extract a function from the given surrogate using the given configuration """
        if type(c) is IndependentExtractionConfig:
            return self.independent_function_extraction(surrogate, c)
        elif type(c) is IndependentIndividualExtractionConfig:
            return self.independent_individual_function_extraction(surrogate, c)
        elif type(c) is WeightedExtractionConfig:
            return self.weighted_function_extraction(surrogate, c)
        else:
            raise ValueError(c)


    def _points_along_y(self, x, ys):
        """ get the points [(x, y) for y in ys] as rows """
        assert len(x.shape) == len(ys.shape) == 2
        assert x.shape[0] == 1 and ys.shape[1] == 1
        xs = np.repeat(x.reshape(1, -1), ys.shape[0], axis=0) # stack copies of x as rows
        return np.hstack((xs, ys))


    def independent_function_extraction(self, surrogate, c):
        """ independent function extraction

        This method is better performing than the 'individual' variant of this
        method because this combines all queries together into a single large
        query. This method exploits the fact that at each control point there is
        a relatively simple 1D optimisation problem which can be approximately
        solved with a reasonable number of samples and doesn't require a local
        optimisation method to solve.
        """
        assert type(c) is IndependentExtractionConfig
        t = Timer()
        extraction_info = Trial.ExtractionInfo()

        if c.sample_distribution == 'linear':
            test_ys = np.linspace(*self.range_bounds, num=c.samples_per_cp).reshape(-1, 1)
        elif c.sample_distribution == 'random':
            test_ys = uniform_random_in_bounds(c.samples_per_cp, [self.range_bounds])
        else:
            raise ValueError(c.sample_distribution)

        X = np.vstack(self._points_along_y(x, test_ys) for x in c.control_xs)
        extraction_info.acq_evals += X.shape[0]
        t_acq = Timer()
        acq = c.acquisition(X, surrogate=surrogate,
                           maximising=self.is_maximising(),
                           return_gradient=False, **c.acquisition_params)
        extraction_info.acq_total_time = t_acq.stop()
        ys = []
        for i in range(len(c.control_xs)):
            best_y = test_ys[np.argmax(acq[i*c.samples_per_cp : (i+1)*c.samples_per_cp])]
            ys.append(best_y)

        ys = np.array(ys).reshape(-1, 1)
        f = InterpolatedFunction(c.control_xs, ys, interpolation=c.interpolation,
                                    clip_range=self.range_bounds if self.clip_range else None)
        extraction_info.duration = t.stop()
        return (f, extraction_info)


    def independent_individual_function_extraction(self, surrogate, c):
        """ independent and individual function extraction

        similar to weighted_function_extraction, this method simply maximises
        the acquisition function at each control point without any weighting
        """
        assert type(c) is IndependentIndividualExtractionConfig
        ys = []
        t = Timer()
        extraction_info = Trial.ExtractionInfo()

        for x in c.control_xs:
            def acq(ys, return_gradient=False):
                X = self._points_along_y(x, ys)
                extraction_info.acq_evals += X.shape[0]
                extraction_info.acq_gradient_evals += X.shape[0] if return_gradient else 0
                t = Timer()
                res = c.acquisition(X, surrogate=surrogate,
                                   maximising=self.is_maximising(),
                                   return_gradient=return_gradient,
                                   **c.acquisition_params)
                extraction_info.acq_total_time += t.stop()
                return res

            best_y, best_acq = c.aux_optimiser(acq, [self.range_bounds], **c.aux_optimiser_params)
            ys.append(np.asscalar(best_y))

        ys = np.array(ys).reshape(-1, 1)
        f = InterpolatedFunction(c.control_xs, ys, interpolation=c.interpolation,
                                    clip_range=self.range_bounds if self.clip_range else None)
        extraction_info.duration = t.stop()
        return (f, extraction_info)


    def weighted_function_extraction(self, surrogate, c):
        """
        Weighted Extraction:
            a hyperplane sweeps through the function input space along the given
            direction and control points are chosen based on maximising an
            acquisition function which is weighted to bias towards the output
            values of the points adjacent to the current point which are behind
            the frontier of the hyperplane.

            This is achieved by a product of experts approach where one expert
            is the acquisition function and another simply wants to stay close
            to previous values by an RBF function centered at the previous
            values.
        """
        assert type(c) is WeightedExtractionConfig
        assert len(self.domain_bounds) == 1 # TODO: support multiple dimensions
        ys = []
        prev_xy = None
        t = Timer()
        extraction_info = Trial.ExtractionInfo()
        for x in c.control_xs:

            X = self._points_along_y(x, np.linspace(*self.range_bounds, num=200).reshape(-1, 1))
            worst_acq = np.min(c.acquisition(X, surrogate=surrogate,
                                             maximising=self.is_maximising(),
                                             **c.acquisition_params))

            def acq(ys, return_gradient=False):
                """ with x fixed, given ys as rows calculate the weighted
                acquisition function for the pairs of x and y
                """
                X = self._points_along_y(x, ys)
                extraction_info.acq_evals += X.shape[0]
                extraction_info.acq_gradient_evals += X.shape[0] if return_gradient else 0
                t = Timer()
                res = c.acquisition(X, surrogate=surrogate,
                                   maximising=self.is_maximising(),
                                   return_gradient=return_gradient,
                                   **c.acquisition_params)
                extraction_info.acq_total_time += t.stop()
                acq, dacq_dx = res if return_gradient else (res, None)
                # shift to make 0 the worst possible value, so that at w=0,
                # A*w is the worst possible value.
                # can't subtract min(acq) because ys may not be many points over the whole domain
                acq -= worst_acq

                #TODO: rather than prev_xy, need get_adjacent() or similar to cope with multiple dimensions
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

        ys = np.array(ys).reshape(-1, 1)
        f = InterpolatedFunction(c.control_xs, ys, interpolation=c.interpolation,
                                    clip_range=self.range_bounds if self.clip_range else None)
        extraction_info.duration = t.stop()
        return (f, extraction_info)


    ################################
    # Fitting Surrogate
    ################################

    def surrogate_dimensionality(self):
        return len(self.domain_bounds) + 1

    def num_available_training_points(self):
        """ the number of training points available for fitting the surrogate
        model. If resampling is used then not all of these points may be used.
        """
        return sum(len(t.R_ls) for t in self.trials)

    def get_training_data(self, c):
        assert type(c) is BayesSelectConfig

        X = [] # (x, y)
        R_l = [] # R_l(x,y)
        for t in self.trials:
            X += [np.append(x, t.f(x)) for x, r in t.R_ls]
            R_l += [r for x, r in t.R_ls]
        X = np.array(X)
        R_l = np.array(R_l).reshape(-1, 1)

        # resample
        if c.resample_method is None or c.resample_num == -1 or X.shape[0] <= c.resample_num:
            pass # do not resample
        elif c.resample_method == 'random':
            ids = np.random.permutation(np.arange(X.shape[0]))[:c.resample_num]
            X, R_l = X[ids], R_l[ids]
            assert X.shape[0] == R_l.shape[0] == c.resample_num
        else:
            raise ValueError(c.resample_method)

        return X, R_l

    def fit_surrogate(self, c):
        """ gather the training data and fit a surrogate model to it

        Args:
            c (BayesSelectConfig): the configuration for the Bayesian trial
        """
        assert type(c) is BayesSelectConfig
        timer = Timer()
        X, R_l = self.get_training_data(c)

        if c.initial_hyper_params == 'last':
            # hyperparameter continuity, use the previous model's hyperparameters as a starting point
            last_params = None
            for t in reversed(self.trials):
                if t.surrogate is not None:
                    last_params = t.surrogate.get_hyper_params()
                    break
            initial_hyper_params = last_params # may be None if this is the first Bayesian trial
        else:
            initial_hyper_params = c.initial_hyper_params # either None or an arbitrary set of parameters

        surrogate = c.surrogate
        surrogate.fit(X, R_l, initial_hyper_params)
        fitting_info = Trial.FittingInfo(timer.stop(), data_set_size=X.shape[0])
        return surrogate, fitting_info

    ################################
    # Bayes Selection
    ################################

    def select_bayes(self, c):
        """
        Args:
            c (BayesSelectConfig): the configuration for the selection
        """
        assert type(c) is BayesSelectConfig
        t = Timer()
        surrogate, fitting_info = self.fit_surrogate(c)
        f, extraction_info = self.extract_function(surrogate, c.extraction_config)
        selection_info = Trial.SelectionInfo(t.stop(), fitting_info, extraction_info)
        return f, surrogate, selection_info


    ################################
    # Bayesian Optimisation Algorithm
    ################################

    def select(self, c):
        """ Select a function using the given configuration
        Args:
            c: the configuration to use for selection
        """
        if type(c) is RandomCPSelectConfig:
            f, selection_info = self.select_random_CP(c)
            surrogate = None
        elif type(c) is GPPriorSelectConfig:
            f, selection_info = self.select_GP_prior(c)
            surrogate = None
        elif type(c) is BayesSelectConfig:
            f, surrogate, selection_info = self.select_bayes(c)
        else:
            raise ValueError(c)

        return f, surrogate, selection_info


    def run(self, coordinator, quiet=False):
        #TODO: rather than taking the OOP approach, could just have this as a stand-alone function and return the list of trials
        assert not self.has_run, 'optimiser already run'
        self.has_run = True

        coordinator.register_optimiser(self)
        timer = Timer()
        trial_num = 0
        while True:
            if not quiet:
                print('trial {}/{}'.format(trial_num+1, coordinator.get_max_trials() or '?'))
            trial_timer = Timer()

            c = coordinator.get_config(trial_num)
            if c is None:
                break # finished

            f, surrogate, selection_info = self.select(c)

            eval_timer = Timer()
            res = self.objective(f)
            if len(res) == 2:
                R_ls, R_g = res
                user_info = None
            elif len(res) == 3:
                R_ls, R_g, user_info = res
            else:
                raise ValueError('wrong number of values ({}) returned from objective function'.format(len(res)))
            evaluation_info = Trial.EvaluationInfo(eval_timer.stop(), user_info)

            t = Trial(trial_num, c, f, R_ls, R_g, surrogate, trial_timer.stop(), selection_info, evaluation_info)
            self.trials.append(t)
            trial_num += 1

        print('optimisation finished: {} trials in {:.1f} seconds'.format(trial_num, timer.stop()))


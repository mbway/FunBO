#!/usr/bin/env python3
"""
Coordination of the optimiser
"""

import numpy as np
import GPy

# local imports
from .utils import FixedAttributes
from . import aux_optimisers



class Coordinator:
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


class DefaultCoordinator(Coordinator):
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
            SelectConfig object => perform a selection using the given configuration
        """
        if trial_num < self.pre_phase_trials:
            return self.get_pre_phase_config(trial_num)
        elif trial_num < self.max_trials:
            return self.get_bayes_config(trial_num)
        else:
            return None

    def get_pre_phase_config(self, trial_num):
        """ override this method to change the behaviour """
        return RandomSelectConfig(self.domain_bounds)

    def get_bayes_config(self, trial_num):
        """ override this method to change the behaviour """
        return BayesSelectConfig(self.domain_bounds)


class GPPriorSelectConfig(FixedAttributes):
    """ Select a random function by sampling from a GP prior

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

        self.fixed_attributes = True # no more attributes

class CPSelectConfig(FixedAttributes):
    """ Select a random function by sampling random control point heights

    Attributes:
        control_xs: the `x` positions of the control points
    """
    def __init__(self, domain_bounds):
        _, xmin, xmax = domain_bounds # TODO: not multi dimensional
        self.control_xs = np.linspace(xmin, xmax, num=10)
        self.interpolation = 'quadratic'

        self.fixed_attributes = True # no more attributes

class BayesSelectConfig(FixedAttributes):
    r"""
    Attributes:
        reward_combo: the weighting $(\alpha, \beta)$ (both $\in[0,1]$) for the
            local rewards $R_l$ and global reward $R_g$ in the convex combination
            $\alpha*R_l + \beta*R_g$
        kernel: the kernel to use for the surrogate model
        surrogate_class: the class to use as the surrogate model, eg `GPy.models.(GPRegression|SparseGPRegression)`
        surrogate_model_params: the parameters to pass to the constructor of surrogate_class
        surrogate_optimise_params: the parameters to pass to optimize_restarts
        surrogate_optimise_init: used for hyperparameter continuity. None => start from a random location
        control_xs: the x values to calculate the optimal y value and use (x,y) as a control point for the function sample
        acquisition: the acquisition function to use
        acquisition_param: the exploration/exploitation trade-off parameter passed to the acquisition function
        tracking_l: the length scale to use in the kernel used in calculating the tracking weights
    """
    def __init__(self, domain_bounds):
        self.reward_combo = (1, 0)
        self.surrogate_class = GPy.models.GPRegression
        #TODO: thin wrapper around GPy
        self.surrogate_model_params = dict(
            kernel=GPy.kern.RBF(input_dim=2, ARD=True), # TODO: not multi-dimensional
            normalizer=True
        )
        self.surrogate_optimise_params = dict(
            parallel=True,
            verbose=False,
            num_restarts=10 # actually the number of iterations
        )
        #TODO
        self.surrogate_init_params = 'last' # for hyperparameter continuity
        _, xmin, xmax = domain_bounds # TODO: not multi dimensional
        self.control_xs = np.linspace(xmin, xmax, num=50)
        self.acquisition = UCB
        self.acquisition_param = 1.0

        self.fixed_attributes = True # no more attributes


class WeightedExtractionConfig(FixedAttributes):
    def __init__(self):
        self.tracking_l = 1.0 #TODO: might want another kernel other than RBF

        # TODO: use this info
        self.aux_optimiser = aux_optimisers.maximise_quasi_Newton
        self.aux_optimiser_params = dict(
            num_random=10_000,
            num_take_random=50,
            num_bfgs=100,
            exact_gradient=False
        )

        self.fixed_attributes = True # no more attributes



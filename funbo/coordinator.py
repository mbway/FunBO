#!/usr/bin/env python3
"""
Coordination of the optimiser
"""

import numpy as np
import GPy

# local imports
from .utils import FixedAttributes, RegularGrid
from . import aux_optimisers
from . import acquisition



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
            RandomCPSelectConfig => perform a random selection with the given configuration
            GPPriorSelectConfig => perform a random selection with the given configuration
            BayesSelectConfig => perform a Bayesian selection with the given configuration
        """
        raise NotImplemented()


class Coordinator(CoordinatorBase):
    """ A coordinator which follows the standard pattern of some number of
    pre_phase trials, then switching to Bayesian optimisation trials for a set
    number of trials.
    """
    def __init__(self, optimiser, pre_phase_trials, max_trials):
        self.optimiser = optimiser
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
        return RandomCPSelectConfig(self.optimiser)

    def get_bayes_config(self, trial_num):
        """ override this method to change the behaviour """
        return BayesSelectConfig(self.optimiser)


class BayesSelectConfig(FixedAttributes):
    r"""
    Attributes:
        surrogate_config: the configuration for fitting the surrogate model
        extraction_config: the configuration for extracting a function from the surrogate
    """
    slots = ('surrogate_config', 'extraction_config')

    def __init__(self, optimiser):
        self.surrogate_config = SurrogateConfig(optimiser)
        self.extraction_config = WeightedExtractionConfig(optimiser)


class GPPriorSelectConfig(FixedAttributes):
    """ Select a random function by sampling from a GP prior

    Attributes:
        mu: the mean function for the Gaussian process prior which is
            sampled from. None => use the zero function
        kernel: the kernel for the Gaussian process prior which is sampled
            from. The type of kernel and its parameters greatly affect the
            behaviour of the sample function.
        control_xs
    """
    slots = ('mu', 'kernel', 'control_xs', 'interpolation')

    def __init__(self, optimiser):
        self.mu = None
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0) # TODO: not multi-dimensional
        self.control_xs = RegularGrid(100, optimiser.domain_bounds)
        self.interpolation = 'linear'


class RandomCPSelectConfig(FixedAttributes):
    """ Select a random function by sampling random control point heights

    Attributes:
        control_xs: the `x` positions of the control points
    """
    slots = ('control_xs', 'interpolation')

    def __init__(self, optimiser):
        self.control_xs = RegularGrid(10, optimiser.domain_bounds)
        self.interpolation = 'cubic'


class SurrogateConfig(FixedAttributes):
    r"""
    Attributes:
        model_class: the class to use as the surrogate model, eg `GPy.models.(GPRegression|SparseGPRegression)`
        init_params: the parameters to pass to the constructor of surrogate_class (see GPy documentation)
        optimise_params: the parameters to pass to optimize_restarts (see paramz documentation)
        initial_hyper_params: The starting point for surrogate hyperparameter
            optimisation. Used for hyperparameter continuity.
            None => start from a random location.
            'last' => use the hyperparameters of the last surrogate as a
                starting point (if there is no previous model then start randomly).
            array => use these hyperparameters as a starting point.
    """
    slots = ('model_class', 'init_params', 'optimise_params', 'initial_hyper_params')

    def __init__(self, optimiser):
        self.model_class = GPy.models.GPRegression
        #TODO: thin wrapper around GPy
        self.init_params = dict(
            kernel=GPy.kern.RBF(input_dim=2, ARD=True), # TODO: not multi-dimensional
            normalizer=True
        )
        self.optimise_params = dict(
            parallel=True, # not usable with SparseGPRegression
            num_processes=None, # => equal to the number of cores
            verbose=False,
            num_restarts=10 # actually the number of iterations
        )
        self.initial_hyper_params = 'last' # for hyperparameter continuity

class IndependentExtractionConfig(FixedAttributes):
    """ Extract a function from the surrogate model by maximising the
        acquisition function at the specified control point locations. At each
        control point, the acquisition function is maximised independently. This
        may cause discontinuities in the function.

    Attributes:
        acquisition: the acquisition function to use
        acquisition_params: additional parameters passed to the acquisition
            function such as an exploration/exploitation trade-off parameter
        aux_optimiser: the method to use when maximising the acquisition function
        aux_optimiser_params: the parameters to pass to aux_optimiser
        control_xs: the x values to calculate the optimal y value and use (x,y) as a control point for the function sample
        interpolation: the interpolation to use between the control points of the extracted function
    """
    slots = (
        'acquisition', 'acquisition_params', 'aux_optimiser',
        'aux_optimiser_params', 'control_xs', 'interpolation'
    )
    def __init__(self, optimiser):
        self.acquisition = acquisition.UCB
        self.acquisition_params = dict(beta=1.0)
        self.aux_optimiser = aux_optimisers.maximise_random_quasi_Newton
        self.aux_optimiser_params = dict(
            num_random=10_000,
            # since extraction consists of many 'easy' 1D optimisations, BFGS is
            # only needed to slightly tweak the best random result.
            num_take_random=1,
            num_bfgs=1,
            exact_gradient=False,
            quiet=True # don't show warnings
        )
        self.control_xs = RegularGrid(50, optimiser.domain_bounds)
        self.interpolation = 'linear'


class WeightedExtractionConfig(IndependentExtractionConfig):
    """ Extract a function from the surrogate model by maximising the
        acquisition function at the specified control point locations, with the
        acquisition function weighted to favour staying close to the adjacent
        output values to prevent discontinuities.

    Attributes:
        tracking_l: the length scale to use in the kernel used in calculating the tracking weights
    """
    slots = IndependentExtractionConfig.slots + (
        'tracking_l',
    )

    def __init__(self, optimiser):
        super().__init__(optimiser) # see IndependentExtractionConfig for the other attributes

        #TODO: tracking_ls (for multiple dimensions)
        self.tracking_l = 1.0 #TODO: might want another kernel other than RBF
        #TODO: growth_directions = [-1, 1, -1] etc to indicate the direction to grow along that dimension



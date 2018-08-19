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
from .surrogates import GPySurrogate



class CoordinatorBase:
    """ Describes a Bayesian optimisation strategy with the implementation
        details abstracted away.

    By inheriting from this class and overriding get_config, the behaviour of
    the optimiser can be completely re-configured.
    """
    def get_max_trials(self):
        """ return the maximum number of trials (can return None if unknown) """
        raise NotImplementedError()

    def register_optimiser(self, optimiser):
        """ called when the coordinator is first used by an optimiser, before any calls to `get_config` """
        raise NotImplementedError()

    def get_config(self, trial_num):
        """ get the selection configuration for this trial

        Returns:
            None => finish the optimisation
            RandomCPSelectConfig => perform a random selection with the given configuration
            GPPriorSelectConfig => perform a random selection with the given configuration
            BayesSelectConfig => perform a Bayesian selection with the given configuration
        """
        raise NotImplementedError()


class Coordinator(CoordinatorBase):
    """ A coordinator which follows the standard pattern of some number of
    pre_phase trials, then switching to Bayesian optimisation trials for a set
    number of trials.
    """
    def __init__(self, pre_phase_trials, max_trials):
        assert pre_phase_trials > 0 and max_trials >= pre_phase_trials
        self.optimiser = None
        self.pre_phase_trials = pre_phase_trials
        self.max_trials = max_trials

    def register_optimiser(self, optimiser):
        self.optimiser = optimiser

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
        surrogate: the `Surrogate` instance to use for the trial
        initial_hyper_params: The starting point for surrogate hyperparameter
            optimisation. Used for hyperparameter continuity.
            None => start from a random location.
            'last' => use the hyperparameters of the last surrogate as a
                starting point (if there is no previous model then start randomly).
            array => use these hyperparameters as a starting point.
        resample_method: the method to use for resampling the training data set
            for the surrogate model. None to disable. 'random' to sample points
            randomly without replacement. Creates a 'subset of data (SoD)'
            approximation to a full GP.
        resample_num: when resample_method is not None, gives the number of
            points to keep in the training data set for the surrogate model.
        extraction_config: the configuration for extracting a function from the surrogate
    """
    slots = ('surrogate', 'initial_hyper_params', 'resample_method', 'resample_num', 'extraction_config')

    def __init__(self, optimiser):
        self.surrogate = GPySurrogate(
            init_params=dict(
                kernel=GPy.kern.RBF(input_dim=optimiser.surrogate_dimensionality(), ARD=False),
                normalizer=True
            ),
            optimise_params=dict(
                parallel=False, # not usable with SparseGPRegression, useless if num_restarts == 1
                num_processes=None, # None => equal to the number of cores
                verbose=False,
                num_restarts=1 # actually the number of iterations
            ),
            sparse=False
        )
        self.initial_hyper_params = 'last' # for hyperparameter continuity
        self.resample_method = None
        self.resample_num = -1
        self.extraction_config = WeightedExtractionConfig(optimiser)


class GPPriorSelectConfig(FixedAttributes):
    """ Select a random function by sampling from a GP prior

    Attributes:
        mu: the mean function for the Gaussian process prior which is
            sampled from. None => use the zero function
        kernel: the kernel for the Gaussian process prior which is sampled
            from. The type of kernel and its parameters greatly affect the
            behaviour of the sample function.
        control_xs: the `x` positions of the control points
    """
    slots = ('mu', 'kernel', 'control_xs', 'interpolation')

    def __init__(self, optimiser):
        self.mu = None
        self.kernel = GPy.kern.RBF(input_dim=len(optimiser.domain_bounds), variance=1.0, lengthscale=1.0)
        self.control_xs = RegularGrid(20, optimiser.domain_bounds)
        self.interpolation = 'linear'


class RandomCPSelectConfig(FixedAttributes):
    """ Select a random function by sampling random control point heights

    Attributes:
        control_xs: the `x` positions of the control points
    """
    slots = ('control_xs', 'interpolation')

    def __init__(self, optimiser):
        self.control_xs = RegularGrid(10, optimiser.domain_bounds)
        self.interpolation = 'linear'


class ExtractionBase(FixedAttributes):
    """ the attributes shared between all function extraction methods

    Attributes:
        acquisition: the acquisition function to use
        acquisition_params: additional parameters passed to the acquisition
            function such as an exploration/exploitation trade-off parameter
        control_xs: the x values to calculate the optimal y value and use (x,y)
            as a control point for the function sample
        interpolation: the interpolation to use between the control points of
            the extracted function
    """
    slots = (
        'acquisition', 'acquisition_params',
        'control_xs', 'interpolation'
    )
    def __init__(self, optimiser):
        self.acquisition = acquisition.UCB
        self.acquisition_params = dict(beta=1.0)
        self.control_xs = RegularGrid(50, optimiser.domain_bounds)
        self.interpolation = 'linear'

class IndependentExtractionConfig(ExtractionBase):
    """ Extract a function from the surrogate model by maximising the
        acquisition function at the specified control point locations.
        Every control point is optimised at the same time by combining queries
        of the acquisition function together. This is done for performance reasons.

    Attributes:
        samples_per_cp: the number of different y values to try for each control point
        sample_distribution: how to choose the y values to try ('random' or 'linear')
    """
    slots = ExtractionBase.slots + ('samples_per_cp', 'sample_distribution')
    def __init__(self, optimiser):
        super().__init__(optimiser) # see ExtractionBase for the other attributes
        self.samples_per_cp = 100
        # anecdotally, linear tends to be a bit smoother
        self.sample_distribution = 'linear' # one of 'random', 'linear'
        #TODO: method = 'random' | 'gradient descent'

class IndependentIndividualExtractionConfig(ExtractionBase):
    """ Extract a function from the surrogate model by maximising the
        acquisition function at the specified control point locations. At each
        control point, the acquisition function is maximised independently. This
        may cause discontinuities in the function.

    Unlike IndependentExtractionConfig, this method is slightly different in
    that it treats each control point as a separate optimisation problem,
    however this has a large (~10x) performance impact.

    Attributes:
        aux_optimiser: the local optimisation method to use when maximising the acquisition function
        aux_optimiser_params: the parameters to pass to aux_optimiser
    """
    slots = ExtractionBase.slots + ('aux_optimiser', 'aux_optimiser_params')
    def __init__(self, optimiser):
        super().__init__(optimiser) # see ExtractionBase for the other attributes
        # because the auxiliary optimiser is only ever optimising along a single
        # dimension at a time, a reasonable number of random samples should be
        # sufficient. BFGS is inadvisable due to the large overhead from
        # querying a GP one point at a time.
        self.aux_optimiser = aux_optimisers.maximise_random_quasi_Newton
        self.aux_optimiser_params = dict(
            num_random=200,
            # since extraction consists of many 'easy' 1D optimisations, BFGS is
            # only needed to slightly tweak the best random result.
            num_take_random=1,
            num_bfgs=0,
            exact_gradient=False,
            quiet=True # don't show warnings
        )

class ElasticExtractionConfig(ExtractionBase):
    """ TODO
    """
    slots = ExtractionBase.slots + ('elastic_stiffness', 'neighbourhood_radius')
    def __init__(self, optimiser):
        super().__init__(optimiser) # see ExtractionBase for the other attributes
        self.elastic_stiffness = 1
        self.neighbourhood_radius = 1
        self.neighbourhood_distribution = 'uniform'


class WeightedExtractionConfig(ExtractionBase):
    """ Extract a function from the surrogate model by maximising the
        acquisition function at the specified control point locations, with the
        acquisition function weighted to favour staying close to the adjacent
        output values to prevent discontinuities.

    Attributes:
        tracking_l: the length scale to use in the kernel used in calculating the tracking weights
        aux_optimiser: the local optimisation method to use when maximising the acquisition function
        aux_optimiser_params: the parameters to pass to aux_optimiser
    """
    slots = ExtractionBase.slots + (
        'aux_optimiser', 'aux_optimiser_params', 'tracking_l', 'sweep_direction'
    )

    def __init__(self, optimiser):
        super().__init__(optimiser) # see ExtractionBase for the other attributes

        # because the auxiliary optimiser is only ever optimising along a single
        # dimension at a time, a reasonable number of random samples should be
        # sufficient. BFGS is inadvisable due to the large overhead from
        # querying a GP one point at a time.
        self.aux_optimiser = aux_optimisers.maximise_random_quasi_Newton
        self.aux_optimiser_params = dict(
            num_random=200,
            # since extraction consists of many 'easy' 1D optimisations, BFGS is
            # only needed to slightly tweak the best random result.
            num_take_random=1,
            num_bfgs=0,
            exact_gradient=False,
            quiet=True # don't show warnings
        )

        #TODO: tracking_ls (for multiple dimensions)
        self.tracking_l = 1.0 #TODO: might want another kernel other than RBF
        #TODO: growth_directions = [-1, 1, -1] etc to indicate the direction to grow along that dimension
        self.sweep_direction = np.zeros(len(optimiser.domain_bounds))
        self.sweep_direction[0] = 1




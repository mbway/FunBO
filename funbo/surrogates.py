#!/usr/bin/env python3

import numpy as np
import GPy
import warnings

from .utils import show_warnings

try:
    from . import distributed_gp as dgp
except ImportError:
    dgp = None


class Surrogate:
    def fit(self, X, y, initial_hyper_params):
        """
        Args:
            X: the inputs to fit to
            y: the targets corresponding to the inputs to fit to
            initial_hyper_params: the hyperparameters to initialise the model
                with (and use as a starting point for any optimisation), None to
                randomize.
        """
        raise NotImplementedError()

    def predict(self, X_new):
        """ predict the mean and variance at the points X_new

        Returns:
            (mu, var) both with shapes == (X_new.shape[0], 1)
        """
        raise NotImplementedError()

    def predict_gradients(self, X_new):
        """ predict the gradients of the mean and variance at the points X_new

        Returns:
            (dmu_dx, dvar_dx) both with shapes == X_new.shape
        """
        raise NotImplementedError()

    def get_hyper_params(self):
        """ get the current model hyperparameters in the form accepted by `initial_hyper_params` in the constructor """
        raise NotImplementedError()

    def get_data_set(self):
        """ get the X, y which the model is fitted to. May be in a different order to what `fit()` was called with """
        raise NotImplementedError()



class GPySurrogate(Surrogate):
    def __init__(self, init_params, optimise_params, sparse=False):
        self.init_params = init_params
        self.optimise_params = optimise_params
        self.sparse = sparse
        self.model = None

    def fit(self, X, y, initial_hyper_params):
        assert self.model is None

        # don't initialise the model until the initial hyperparameters have been set
        # will always raise RuntimeWarning("Don't forget to initialize by self.initialize_parameter()!")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*initialize_parameter.*')
            gp_class = GPy.models.SparseGPRegression if self.sparse else GPy.models.GPRegression
            self.model = gp_class(X, y, initialize=False, **self.init_params)

        # these steps for initialising a model from stored parameters are from https://github.com/SheffieldML/GPy
        self.model.update_model(False)  # prevents the GP from fitting to the data until we are ready to enable it manually
        self.model.initialize_parameter()  # initialises the hyperparameter objects

        if initial_hyper_params is None:
            self.model.randomize()
        else:
            self.model[:] = initial_hyper_params

        self.model.update_model(True)

        # the current parameters are used as one of the starting locations (as of the time of writing)
        # https://github.com/sods/paramz/blob/master/paramz/model.py
        with warnings.catch_warnings(record=True) as ws:
            # num_restarts is actually the number of iterations
            r = self.optimise_params.get('num_restarts', None)
            if r is None or r > 0:
                if self.sparse and self.optimise_params.get('parallel', True):
                    raise Exception('cannot optimise sparse GP in parallel due to a bug: https://github.com/SheffieldML/GPy/issues/651')

                #if self.sparse and 'Z' not in self.init_params:
                #    # if the inducing points (Z) were not specified in the
                #    # constructor then they are chosen randomly from the data
                #    # set and by default _not_ included in the optimisation.
                #    # see https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/sparse_gp_regression.ipynb
                #    # by unconstraining, they are optimised (which increases training time)
                #    self.model.randomize()
                #    self.model.Z.unconstrain()

                self.model.optimize_restarts(**self.optimise_params)

        if self.optimise_params.get('verbose', False):
            show_warnings(ws)

    def predict(self, X_new):
        assert self.model is not None
        assert X_new.ndim == 2
        # full_cov: whether to return the full covariance matrix or just the diagonal
        mu, var = self.model.predict(X_new, full_cov=False)
        var = np.clip(var, 1e-10, np.inf) # ensure no negative variance (breaks sqrt)
        # both have shapes (input_rows, outputs). Since not using a multi-out GP this means (input_rows, 1)
        assert mu.shape == var.shape == (X_new.shape[0], 1)
        return mu, var

    def predict_gradients(self, X_new):
        assert self.model is not None
        assert X_new.ndim == 2
        dmu_dx, dvar_dx = self.model.predictive_gradients(X_new)
        # dmu_dx has shape = (input_rows, input_cols, outputs)
        dmu_dx = dmu_dx[:, :, 0] # take only the first output since not using a multi-out GP
        # all outputs have the same variance so dvar_dx has shape (input_rows, input_cols)
        assert dmu_dx.shape == dvar_dx.shape == X_new.shape
        return dmu_dx, dvar_dx

    def get_hyper_params(self):
        assert self.model is not None
        return np.copy(self.model[:])

    def get_data_set(self):
        assert self.model is not None
        return (self.model.X, self.model.Y)

#TODO
#class ScikitSurrogate(Surrogate):
#    def __init__(self, init_params, optimise_params):
#        self.init_params = init_params
#        self.optimise_params = optimise_params
#        self.model = False



class DistributedGPSurrogate:
    def __init__(self, init_params, optimise_params, predict_params=None, parallel=False):
        assert dgp is not None, 'distributed_gp module not found. Cannot use DistributedGPSurrogate.'
        self.init_params = init_params
        self.optimise_params = optimise_params
        self.predict_params = predict_params or {}
        self.parallel = parallel
        self.model = None

    def fit(self, X, y, initial_hyper_params):
        assert self.model is None
        gp_class = dgp.ParallelDistributedGP if self.parallel else dgp.DistributedGP
        self.model = gp_class(X, y, **self.init_params)
        if initial_hyper_params is not None:
            self.model.set_theta(initial_hyper_params)
        self.model.optimise_params(**self.optimise_params)

    def predict(self, X_new):
        assert self.model is not None
        assert X_new.ndim == 2
        mu, var = self.model.predict(X_new, **self.predict_params)
        var = np.clip(var, 1e-10, np.inf) # ensure no negative variance (breaks sqrt)
        # both have shapes (input_rows, outputs). Since not using a multi-out GP this means (input_rows, 1)
        assert mu.shape == var.shape == (X_new.shape[0], 1)
        return mu, var

    def predict_gradients(self, X_new):
        raise NotImplementedError()

    def get_hyper_params(self):
        assert self.model is not None
        return np.copy(self.model.theta)

    def get_data_set(self):
        assert self.model is not None
        X = np.vstack([e.model.X for e in self.model.experts])
        Y = np.vstack([e.model.Y for e in self.model.experts])
        return (X, Y)



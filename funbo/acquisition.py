#!/usr/bin/env python3
"""
Acquisition Functions
"""

import numpy as np

# local imports
from .utils import k_RBF


def UCB(X, beta, surrogate, maximising, return_gradient=False):
    """
    Args:
        X: the inputs for the surrogate to predict at (one per row)
        beta: the exploration-exploitation trade-off parameter
        surrogate: the surrogate model to use for predictions
        maximising: whether the optimiser is maximising or minimising (minimising => -LCB rather than UCB)
    """
    assert len(X.shape) == 2

    # full_cov: whether to return the full covariance matrix or just the diagonal
    mu, var = surrogate.predict(X, full_cov=False)
    assert mu.shape == var.shape == (X.shape[0], 1) # (input_rows, outputs)

    sigma = np.sqrt(np.clip(var, 1e-10, np.inf)) # ensure no negative variance

    sf = 1 if maximising else -1 # scale factor
    # in this form it is clearer that the value is the negative LCB when minimising
    # sf * (mu + sf * beta * sigma)
    acq = sf * mu + beta * sigma

    if not return_gradient:
        return acq
    else:
        dmu_dx, dvar_dx = surrogate.predictive_gradients(X)
        assert dmu_dx.shape == (X.shape[0], X.shape[1], 1) # (input_rows, input_cols, outputs)
        assert dvar_dx.shape == X.shape # all outputs have the same variance

        dmu_dx = dmu_dx[:, :, 0] # take only the first output
        assert dmu_dx.shape == X.shape

        # sigma must be strictly > 0 for this, which has been ensured earlier
        #
        # $\frac{d\sigma}{dx} = \frac{d\sigma}{d\sigma^2} \frac{d\sigma^2}{dx}$
        # $=\frac{1}{\frac{d\sigma^2}{d\sigma}} \frac{d\sigma^2}{dx}$
        #
        # uses the 'inverse function theorem'
        # https://math.stackexchange.com/q/185004
        # https://math.stackexchange.com/q/292590
        #
        # by not totally rigorous proof using the chain rule:
        # $\frac{dx}{dy}\frac{dy}{dx}=\frac{dx}{dx}=1 \quad\implies\quad \frac{dx}{dy}=\frac{1}{\frac{dy}{dx}}$
        # requires that dx_dy and dy_dx are > 0
        dsig_dx = dvar_dx / (2*sigma)

        dacq_dx = sf * dmu_dx + beta * dsig_dx

        return acq, dacq_dx




def RBF_weighted(X, Y, dY_dx, center, sigma, l):
    """ weight the Y values based on the distance of the corresponding X values to the center of an RBF function with length scale l

    Args:
        X: the input points (as rows) for the Y values to weight
        Y: the output values to weight
        dY_dx: the derivative of Y w.r.t X. None => don't calculate weighted gradient.
        center: the center of the RBF to compare with each point in X
        l: the length scale of the RBF

    Returns:
        `weighted_Y if dY_dx is None else (weighted_Y, dweightedY_dx)`
    """
    assert Y.shape == (X.shape[0], 1)
    assert (dY_dx is None) or (dY_dx.shape == X.shape)
    assert center.shape == (1, X.shape[1])
    assert np.isscalar(sigma) and np.isscalar(l)
    # for the input points X, calculate the weight using an RBF function
    return_gradient = bool(dY_dx is not None)
    res = k_RBF(X, center, sigma=sigma, l=l, return_gradient=return_gradient)
    w, dw_dx = res if return_gradient else (res, None)
    weighted_Y = Y * w
    if return_gradient:
        # product rule
        dweightedY_dx = Y * dw_dx + dY_dx * w
        return weighted_Y, dweightedY_dx
    else:
        return weighted_Y


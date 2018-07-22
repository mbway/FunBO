#!/usr/bin/env python3
"""
Acquisition Functions
"""

import numpy as np


def UCB(X, beta, surrogate, maximising):
    """
    Args:
        X: the inputs for the surrogate to predict at (one per row)
        beta: the exploration-exploitation trade-off parameter
        surrogate: the surrogate model to use for predictions
        maximising: whether the optimiser is maximising or minimising (minimising => -LCB rather than UCB)
    """
    mu, var = surrogate.predict(X)
    assert mu.shape == var.shape == (X.shape[0], 1)
    sigma = np.sqrt(np.clip(var, 0, np.inf)) # ensure no negative variance
    sf = 1 if maximising else -1 # scale factor
    # in this form it is clearer that the value is the negative LCB when minimising
    # sf * (mus + sf * beta * sigmas)
    return sf * mu + beta * sigma


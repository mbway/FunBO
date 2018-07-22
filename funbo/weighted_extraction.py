#!/usr/bin/env python3
"""
Extract a function from the surrogate using the weighted extraction method
"""


def k_RBF(r, sigma, l):
    return sigma**2 * np.exp(-0.5 * np.square(r/l))


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

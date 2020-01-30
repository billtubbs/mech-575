"""
Implementation of SINDy algorithm based on code by Steven L.
Brunton.  See Paper, 'Discovering Governing Equations from Data:
Sparse Identification of Nonlinear Dynamical Systems' by S. L.
Brunton, J. L. Proctor, and J. N. Kutz.
Original code available from: https://www.eigensteve.com
"""
import numpy as np


def polynomial_features(y_in, order=3):
    """Calculate polynomial terms up to given order for
    all data points in y_in.  This function is similar
    to sklearn.preprocessing.PolynomialFeatures method
    but considerably faster.

    Args:
        y_in (array): m x n array containing m data points 
            for n input variables.
        poly_order (int): Order of polynomial to generate
            terms for (1, 2 or 3).
    
    Returns:
        y_out (array): 
    """
    if len(y_in.shape) == 1:
        y_in = y_in.reshape(-1, 1)  # Reshape vector to matrix
    n = y_in.shape[1]
    y_out_cols = []

    # Poly order 0
    y_out_cols.append(np.ones((len(y_in), 1)))

    # Poly order 1
    y_out_cols.append(y_in)

    # Poly order 2
    if order >= 2:
        for i in range(n):
            y_out_cols.append(y_in[:, i:] * y_in[:, i].reshape(-1, 1))

    # Poly order 3
    if order >= 3:
        # Use poly order 2 results
        results = y_out_cols[-n:]
        for j in range(0, n):
            for result in results[j:]:
                y_out_cols.append(result * y_in[:, j].reshape(-1, 1))

    return np.hstack(y_out_cols)


def sparsify_dynamics_lstsq(theta, dxdt, lamb, n):
    """SINDy algorithm to find sparse polynomial model
    of dynamics using ordinary least-squares.
    """

    # Initial guess: Least-squares
    xi = np.linalg.lstsq(theta, dxdt, rcond=None)[0]

    for k in range(10):
        # Find small coefficients below threshold
        smallinds = np.abs(xi) < lamb
        xi[smallinds] = 0
        for ind in range(n):  # n is state dimension
            biginds = np.logical_not(smallinds[:, ind])
            # Regress dynamics onto remaining terms to find sparse xi
            xi[biginds, ind] = np.linalg.lstsq(theta[:, biginds], dxdt[:, ind],
                                               rcond=None)[0]

    return xi

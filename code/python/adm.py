# Python implementation of Alternating Direction Method
# Adapted from MATLAB version by Niall Mangan from here:
# https://github.com/niallmm/Hybrid-SINDy


import numpy as np
import matplotlib.pyplot as plt
from optht import optht


def null(a, rtol=1e-5):
    """Computes the null space of a matrix, similar to MATLAB's
    null function. Returns an orthonormal basis for the null
    space of A.
    """
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()


def adm(y, q_init, lam, max_iter, tol):
    q = q_init
    for k in range(max_iter):
        q_old = q

        # Update y by soft thresholding
        x = soft_thresholding(y*q, lam)

        # Update
        #  q by projection to the sphere
        q = y.T.dot(x) / np.linalg.norm(y.T.dot(x), ord=2)
        res_q = np.linalg.norm(q_old - q, ord=2)
        if res_q <= tol:
            break

    return q


def soft_thresholding(X, d):
    """soft-thresholding operator
    """
    # return np.maximum(np.abs(X) - d, 0)
    return np.sign(X) * np.maximum(np.abs(X) - d, 0)


def adm_pareto(theta, tol, plot=False):
    """Calculates the nullspace of theta with noise.

    Args
        theta (array): 
        tol (float): Tolerance
        plot (bool or int): Show plots if plot is True or > 0.

    Uses Donoho optimal shrinkage code to find the correct threshold for the
    singular values in the presence of noise.
    Uses ADM algorithm to compute the linear combination of basis vectors
    spanning the nullspace of Theta that creates the sparsest resulting
    vector. This sparse vector gives the coefficients Xi so that Theta*Xi = 0
    and therefore select the terms in a sparse model.

    If the plot flag is True or 1, it plots the Pareto front
    If plot=2 then it will plot some diagnostic plots including the number
    of terms for each attempted initial condition. If you have some initial
    conditions that result in fairly sparse vectors, you can decrease the
    tolerance to improve the "resolution" of these from next best initial
    condtions.

    Returns: 
        xi, ind_theta, lambda_vec, num_terms, error_v

    Where:
        - xi is a vector of the coefficients for the nonzero terms at each lambda
        - ind_theta is a cell containing the indicies for the nonzero terms in
            theta for each value of lambda tried.
        - lambda_vec is a lambda vector with the number of lambda values tried for that 
        variable's set of data
        - num_terms is a vector of the number of terms for each lambda value tried.
    """

    xi = np.zeros((theta.shape[1], 1))
    ind_theta = []

    # initial lambda value, which is the value used for soft thresholding
    # in ADM
    lam = 1e-8

    # counter
    jj = 1

    # initialize the number of nonzero terms found for the lambda
    num = 1
    max_iter = 1e4

    # Use Donoho optimal shrinkage code to find null space in presence of
    # noise.
    U, S, Vh = np.linalg.svd(theta.T, full_matrices=False)
    m, n = theta.T.shape

    # m/n aspect ratio of matrix to be denoised
    ydi = np.diagonal(theta.T).copy()
    ydi[ydi < (optht(m / n, sigma=False) * np.median(ydi))] = 0
    theta2 = U.dot(np.diag(ydi)).dot(Vh).T
    null_theta = null(theta2)

    # Vary lambda by factor of 2 until we hit the point
    # where all coefficients are forced to zero.
    # could also use bisection as commented out below
    while num > 0:
        print(jj)
        # Use ADM algorithm with varying intial conditions to find coefficient
        # matrix
        ind_theta1, xi1, num_terms1 = adm_initvary(null_theta, lam, max_iter, tol, plot=plot)
        
        # Save the indices of non zero coefficients
        ind_theta.append(ind_theta1)
        xi[:, jj] = xi1  # get those coefficients

        # Calculate how many terms are in the sparsest vector
        num_terms[jj, 0] = num_terms1
    
        # Calculate the error for the sparse vector found given this lambda
        error_v[jj, 0] = np.sum(theta.dot(xi[:,jj]))
        # Store
        lambda_vec[jj, 0] = lam
        # Index
        lam = 2 * lam
        num = num_terms[jj, 1]
        jj = jj + 1

    if plot > 0:
        plt.figure(33)
        plt.semilogy(num_terms, np.abs(error_v), 'o')
        plt.xlabel('Number of terms')
        plt.ylabel('Error')
        plt.title('Pareto Front')

        plt.figure(34)
        plt.loglog(lambda_vec, num_terms, 'o') 
        plt.xlabel('Lambda values')
        plt.ylabel('Number of terms')
        plt.show()

    return xi, ind_theta, lambda_vec, num_terms, error_v


def adm_initvary(null_theta, lam, max_iter, tol, plot=False):

    # normalize the columns of the null space of theta to
    # use in the initial condition for ADM search.
    # TODO: Is this correct?  Shouldn't it be divided by std dev.?
    # Original MATLAB code:
    # for ii = 1:size(nT,2)
    #     nTn(:,ii) = nT(:,ii)/mean(nT(:,ii));
    # end
    nTn = null_theta / null_theta.mean()

    # Run ADM algorithm on each row of nTn
    for jj in range(nTn.shape[0]):
        print(jj)
        # initial conditions
        qinit = nTn[jj, :].T

        # Algorithm for finding coefficients resulting in
        # sparsest vector in null space
        q[:, jj] = adm(null_theta, qinit, lam, max_iter, tol)

        # Compose sparsets vectors
        out[:, jj] = null_theta * q[:,jj]

        # Check how many zeros each of the found sparse vectors have
        n_zeros[jj] = (out[:, jj].abs() < lam).sum()

    # Find the vector with the largest number of zeros
    ind_sparse = np.argmax(n_zeros)

    # Save the indices of non zero coefficients
    ind_theta = np.where(out[:, ind_sparse[0].abs()] >= lam)
    # Xi = out[indTheta, indsparse(1)]  # get those coefficients

    # Get sparsest vector
    xi = out[:, ind_sparse[0]]
    small_inds = out[:, ind_sparse[1]].abs() < lam

    # Set thresholded coefficients to zero
    xi[small_inds] = 0

    # check that the solution found by ADM is unique.
    if len(ind_sparse) > 1:
        xi_diff = out[ind_theta, ind_sparse[0]] - out[ind_theta, ind_sparse[1]]
        if xi_diff > tol:
            print('WARNING: ADM has discovered two different sparsest vectors')
    # calculate how many terms are in the sparsest vector
    num_terms = len(ind_theta)

    if plot:
        plt.figure(121)
        plt.semilogy(nT.shape[0] - n_zeros, 'o')
        plt.show()
    return ind_theta, xi, num_terms


if __name__ == '__main__':

    # Run some tests
    a = np.array([
        [16,   2,     3,    13],
        [5,   11,    10,     8],
        [9,    7,     6,    12],
        [4,   14,    15,     1]
    ])
    null_a_calc = null(a)
    null_a_true = np.array([
        0.2236, 0.6708, -0.6708, -0.2236
    ]).reshape(-1, 1)
    assert np.allclose(null_a_calc, null_a_true, atol=0.0001)

    y = np.array([
        [0.8147,    0.0975],
        [0.9058,    0.2785],
        [0.1270,    0.5469],
        [0.9134,    0.9575],
        [0.6324,    0.9649]
    ])
    lam = 0.2
    max_iter = 1
    tol = 0.0001
    q = 2

    x = soft_thresholding(y*q, lam)
    x_test = np.array([
        [1.4294,       0],
        [1.6116,  0.3570],
        [0.0540,  0.8938],
        [1.6268,  1.7150],
        [1.0648,  1.7298]
    ])
    assert np.allclose(x, x_test, atol=0.0001)
    
    adm_actual = adm(y, 2, lam, max_iter, tol)
    adm_target = np.array([
        [0.6365,  0.4115],
        [0.4255,  0.5181]
    ])
    assert np.allclose(adm_actual, adm_target, atol=0.0001)

    theta = np.random.randn(10,8)
    adm_pareto(theta, 1e-4, 0)

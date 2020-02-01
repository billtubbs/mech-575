# Python implementation of Alternating Direction Method
# Adapted from MATLAB version by Niall Mangan from here:
# https://github.com/niallmm/Hybrid-SINDy


import numpy as np
import matplotlib.pyplot as plt
from optht import optht
from adminitvary.py import ADMinitvary


def null(a, rtol=1e-5):
    """Computes the null space of a matrix, similar to MATLAB's
    null function. Returns an orthonormal basis for the null
    space of A.
    """
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()


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
    ind_theta = cell(1, 1)  # cell array of empty matrices

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
    ydi = np.diag(theta.T)
    ydi[ydi < (optht(m / n, sigma=False) * np.median(ydi))] = 0
    theta2 = (U * np.diag(ydi) * Vh).T
    n_theta = null(theta2)

    # Vary lambda by factor of 2 until we hit the point
    # where all coefficients are forced to zero.
    # could also use bisection as commented out below
    while num > 0:
        print(jj)
        # Use ADM algorithm with varying intial conditions to find coefficient
        # matrix
        ind_theta1, xi1, num_terms1 = adm_initvary(nT, lam, max_iter, tol, plot=plot)
        
        # Save the indices of non zero coefficients
        ind_theta[jj, 0] = ind_theta1
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


def adm_initvary(nT, lam, max_iter, tol, plot=False):

    # normalize the collumns of the null space of Theta to use in the
    # initial conditoin for ADM search
    for ii in range(nT.shape[1]):
        nTn[:, ii] = nT[:, ii] / np.mean(nT[:, ii])

    # run ADM algorithm on each row of nTn
    for jj in range(nTn.shape[1]):
        print(jj)
        q_init = nTn[jj, :].T  # intial conditions
        q[:, jj] = adm(nT, q_init, lam, max_iter, tol)  # algrorithm for
        # finding coefficients resutling in  sparsest vector in null space
        out[:, jj] = nT * q[:, jj]  # compose sparsets vectors
        n_zeros[jj] = length(find(np.abs(out[:, jj]) < lam))  # chech how many zeros each
        # of the found sparse vectors have

    ind_sparse = find(n_zeros == np.max(n_zeros))  # find the vector with the largest number of zeros
    ind_theta = find(np.abs(out[:, ind_sparse[0]]) >= lam)   # save the indices of non zero coefficients
    # xi = out[ind_theta, ind_sparse[0]]  # get those coefficients
    xi = out[:, ind_sparse[0]]  # get sparsest vector
    small_inds = np.abs(out[:, ind_sparse[0]]) < lam
    xi(small_inds) = 0  # set thresholded coefficients to zero


    # check that the solution found by ADM is unique.
    if length(indsparse) > 1:
        xi_diff = out[ind_theta, ind_sparse[0]] - out[ind_theta, indsparse[1]]
        if xi_diff > tol
            print('WARNING: ADM has discovered two different sparsest vectors')
    # calculate how many terms are in the sparsest vector
    numterms = length(indTheta)

    if plot:
        plt.figure(121)
        plt.semilogy(size(nT, 1) - nzeros, 'o')
        plt.show()
    return ind_theta, xi, num_terms


if __name__ == '__main__':
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
    ADMpareto(theta, 1e-4, 0)

# Python implementation of Alternating Direction Method
import numpy as np
import matplotlib.pyplot as plt


def null(a, rtol=1e-5):
    """Compute the null space of a.  Similar to MATLAB's null function,
    returns an orthonormal basis for the null space of A.
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
        if (res_q <= tol):
            break

    return q


def soft_thresholding(X, d):
    """soft-thresholding operator
    """
    # return np.maximum(np.abs(X) - d, 0)
    return np.sign(X) * np.maximum(np.abs(X) - d, 0)


def ADMpareto(theta, tol, pflag):

    xi = np.zeros((theta.shape[1], 1))
    ind_theta = cell(1, 1)  # cell array of empty matrices

    # initial lambda value, which is the value used for soft thresholding in ADM
    lam = 1e-8

    # counter
    jj = 1

    # initialize the number of nonzero terms found for the lambda
    num= 1
    MaxIter = 1e4

    # use Donoho optimal shrinkage code to find null space in presence of
    # noise.
    U, S, Vh = np.linalg.svd(theta.T, 'econ')
    m, n = theta.T.shape

    # m/n aspect ratio of matrix to be denoised
    ydi = np.diag(theta.T)
    ydi[ydi < (optimal_SVHT_coef(m / n, 0) * median(ydi))] = 0
    theta2 = (U * np.diag(ydi) * V.T).T
    nT = null(theta2)

    # vary lambda by factor of 2 until we hit the point
    # where all coefficients are forced to zero.
    # could also use bisection as commented out below
    while num > 0:
        jj
        # use ADM algorithm with varying intial conditions to find coefficient
        # matrix
        ind_theta1, xi1, num_terms1 = ADMinitvary(nT, lam, max_iter, tol, pflag)
        
        ind_theta[jj, 0] = ind_theta1  # save the indices of non zero coefficients
        xi[:, jj] = xi1  # get those coefficients
        num_terms[jj, 0] = num_terms1  # calculate how many terms are in the sparsest vector
    
        # calculate the error for the sparse vector found given this lambda
        errorv[jj, 0] = np.sum(theta.dot(xi[:,jj]))
        # store
        lambda_vec[jj, 0] = lam
        # index
        lam = 2 * lam
        num = numterms[jj, 1]
        jj = jj + 1


    if pflag > 0:
        plt.figure(33)
        plt.semilogy(numterms, abs(errorv), 'o')
        plt.xlabel('Number of terms')
        plt.ylabel('Error')
        plt.title('Pareto Front')
        
        plt.figure(34)
        plt.loglog(lambdavec, numterms, 'o') 
        plt.xlabel('Lambda values')
        plt.ylabel('Number of terms')
        plt.show()

    return xi, ind_theta, lambda_vec, num_terms, error_v


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

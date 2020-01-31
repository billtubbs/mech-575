import numpy as np
import matplotlib.pyplot as plt
from adm import adm


def ADMinitvary(nt, lam, max_iter, tol, plotflag):
    """normalize the collumns of the null space of theta to
    use in the initial condition for ADM search.
    """
    for ii in range(nt.shape[1]):
        nTn[:, ii] = nt[:, ii] / np.mean(nt[:, ii])
    
    # Run ADM algorithm on each row of nTn
    for jj in range(nTn.shape[0]):
        print(jj)
        # initial conditions
        qinit = nTn[jj, :].T

        # Algorithm for finding coefficients resulting in
        # sparsest vector in null space
        q[:,jj] = adm(nt, qinit, lam, max_iter, tol)

        # Compose sparsets vectors
        out[:, jj] = nt * q[:,jj]

        # Check how many zeros each of the found sparse vectors have
        nzeros[jj] = len(find(np.abs(out[:, jj]) < lam))

    # Find the vector with the largest number of zeros
    indsparse = find(nzeros == max(nzeros))

    # Save the indices of non zero coefficients
    indTheta = find(abs(out[:, indsparse[0]) >= lam)
    # Xi = out[indTheta, indsparse(1)]  # get those coefficients

    # Get sparsest vector
    xi = out[:, indsparse[0]]
    smallinds = (abs(out(:, indsparse[1])) < lam)

    # Set thresholded coefficients to zero
    xi[smallinds] = 0

    # check that the solution found by ADM is unique.
    if len(ind_sparse) > 1:
        Xidiff= (out[indTheta, ind_sparse[0]] - out[indTheta, ind_sparse[1]]);
        if Xidiff > tol:
            warning('ADM has discovered two different sparsest vectors')

    # calculate how many terms are in the sparsest vector
    num_terms = len(indTheta)

    if plotflag == 2:
        plt.figure(121)
        plt.semilogy(nT.shape[0] - nzeros, 'o')
        plt.show()

    return indTheta, xi, num_terms
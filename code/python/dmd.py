import numpy as np
from numpy.lib.stride_tricks import as_strided

# Note: When translating from MATLAB:
# a\b can be replaced by linalg.solve(a,b) if a is square,
# or linalg.lstsq(a,b) otherwise


def dmd(x, x_prime, r):
    """Compute dynamic matrix decomposition.
    """

    # Step 1
    u, sigma, vh = np.linalg.svd(x, full_matrices=False)
    
    u_r = u[:, :r]
    sigma_r = np.diag(sigma[:r])
    v_r = vh[:r, :]

    # Step 2
    a_tilde = np.linalg.solve(sigma_r.T, (u_r.conj().T.dot(x_prime).dot(v_r.conj().T).T).T)
    
    # Step 3
    lam, w = np.linalg.eig(a_tilde) 

    # Step 4
    phi = x_prime.dot(np.linalg.solve(sigma_r.T, v_r).T).dot(w)
    alpha1 = sigma_r.dot(v_r[:, 0])    
    b = np.linalg.solve(w.dot(np.diag(lam)), alpha1)
    
    return phi, lam, b


def dmd_SB(X, Xprime, r):
    """Compute dynamic matrix decomposition.
    """

    #Code from Brunton & Kutz's book.
    U, Sigma,VT = np.linalg.svd(X, full_matrices=0) # Step 1
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    Atilde = np.linalg.solve(Sigmar.T, (Ur.T @ Xprime @ VTr.T).T).T # Step 2
    Lambda, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(Lambda)
    
    Phi = Xprime @ np.linalg.solve(Sigmar.T, VTr).T @ W # Step 4
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda, alpha1)
    return Phi, Lambda, b

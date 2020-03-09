import numpy as np
from numpy.lib.stride_tricks import as_strided

# When translating from MATLAB:
# a\b
# Can be replaced by linalg.solve(a,b) if a is square,
# linalg.lstsq(a,b) otherwise


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
    # a_tilde2 = u_r.conj().T.dot(x_prime).dot(v_r).dot(np.linalg.inv(sigma_r))
    # assert np.array_equal(a_tilde, a_tilde2)
    
    # Step 3
    lam, w = np.linalg.eig(a_tilde) 

    # Step 4
    phi = x_prime.dot(np.linalg.solve(sigma_r.T, v_r).T).dot(w)
    alpha1 = sigma_r.dot(v_r[:, 0])    
    b = np.linalg.solve(w.dot(np.diag(lam)), alpha1)
    
    return phi, lam, b


def DMD_SB(X, Xprime, r):
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


def dmd_other(X, Y, truncate=None):
    U2, Sig2, Vh2 = np.linalg.svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:, :r]
    Sig = np.diag(Sig2)[:r, :r]
    V = Vh2.conj().T[:, :r]
    Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.inv(Sig)) # build A tilde
    mu, W = np.linalg.eig(Atil)
    Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W) # build DMD modes1
    return mu, Phi


def hankel_matrix(y, m, n):
    """Compute the Hankel matrix from impulse response data.
    
    Arguments
        y : Impulse response data as either array of shape (nt, )
            for SISO system, or (nt, nout, nin) for MIMO system.
        m, n : Dimensions of Hankel matrix (nt >= m + n - 1).
    
    Returns
        h : Hankel matrix as array of shape (m*q, n*p).
    """
    nt = y.shape[0]
    assert nt >= m + n - 1
    if len(y.shape) == 1:
        y = y.reshape(nt, 1, 1)
        p = q = 1  # SISO system
    elif len(y.shape) == 3:
        q, p = y.shape[1:3]  # MIMO system (nout, nin)
    else:
        raise ValueError("y must be 1- or 3-dimensional")
    s0, s1, s2 = y.strides
    return as_strided(y, shape=(m, q, n, p), 
                      strides=(s0, s2, s0, s1),
                      writeable=False).reshape(m*q, n*p)


def hankel_matrix_loops(y, m, n):
    """Compute the Hankel matrix from impulse response data.
    
    Arguments
        y : Impulse response data as array of shape (nout, nin, nt).
        m, n : Dimensions of Hankel matrix (nt >= m + n - 1).
    
    Returns
        h : Hankel matrix as m x n array.
    """
    nt = y.shape[2]
    assert nt >= m + n - 1
    q, p = y.shape[0:2]  # number of outputs, inputs
    h = np.zeros((q*m, p*n), dtype=y.dtype)
    for i in range(m):
        for j in range(n):
            h[q*i:q*(i+1), p*j:p*(j+1)] = y[:, :, i+j]
    return h


def unit_tests():

    # Test hankel_matrix
    y = np.arange(5)
    h = hankel_matrix(y, 3, 3)
    h_true = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4]
    ])
    assert np.array_equal(h, h_true)

    y = np.arange(5).astype(float)
    h = hankel_matrix(y, 4, 2)
    h_true = np.array([
        [0., 1.],
        [1., 2.],
        [2., 3.],
        [3., 4.]
    ])
    assert np.array_equal(h, h_true)

    y = np.arange(20).reshape([5,2,2])
    h = hankel_matrix(y, 3, 3)
    h_true = np.array([
        [ 0,  2,  4,  6,  8, 10],
        [ 1,  3,  5,  7,  9, 11],
        [ 4,  6,  8, 10, 12, 14],
        [ 5,  7,  9, 11, 13, 15],
        [ 8, 10, 12, 14, 16, 18],
        [ 9, 11, 13, 15, 17, 19]
    ])
    assert np.array_equal(h, h_true)

    y = np.arange(30).reshape([5,2,3])
    h = hankel_matrix(y, 2, 4)
    h_true = np.array([
        [ 0,  3,  6,  9, 12, 15, 18, 21],
        [ 1,  4,  7, 10, 13, 16, 19, 22],
        [ 2,  5,  8, 11, 14, 17, 20, 23],
        [ 6,  9, 12, 15, 18, 21, 24, 27],
        [ 7, 10, 13, 16, 19, 22, 25, 28],
        [ 8, 11, 14, 17, 20, 23, 26, 29]
    ])
    assert np.array_equal(h, h_true)

    yT = np.arange(30).reshape([5,2,3])  # Excess data
    y = np.transpose(yT, axes=(1, 2, 0))
    h = hankel_matrix(yT, 2, 3)
    h_true = np.array([
        [ 0,  3,  6,  9, 12, 15],
        [ 1,  4,  7, 10, 13, 16],
        [ 2,  5,  8, 11, 14, 17],
        [ 6,  9, 12, 15, 18, 21],
        [ 7, 10, 13, 16, 19, 22],
        [ 8, 11, 14, 17, 20, 23]
    ])
    assert np.array_equal(h, h_true)
    
    # Arbitrary sized time-series data
    q = 3  # Number of inputs
    p = 2  # Number of outputs
    nt = 10  # Number of timesteps
    m = n = 5  # Hankel matrix dimensions
    yT = np.arange(nt*p*q).reshape(nt,p,q)
    y = np.transpose(yT, axes=(1, 2, 0))
    assert y.shape == (p, q, nt)
    h1 = hankel_matrix_loops(y, m, n)
    h2 = hankel_matrix(yT, m, n)
    assert np.array_equal(h1, h2)


if __name__ == "__main__":
    unit_tests()
import numpy as np

# When translating from MATLAB:
# a\b
# Can be replaced by linalg.solve(a,b) if a is square,
# linalg.lstsq(a,b) otherwise

def dmd(x, x_prime, r):

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


def DMD_SB(X,Xprime,r):
    U,Sigma,VT = np.linalg.svd(X,full_matrices=0) # Step 1
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T).T).T # Step 2
    Lambda, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(Lambda)
    
    Phi = Xprime @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda,alpha1)
    return Phi, Lambda, b


def dmd_other(X, Y, truncate=None):
    U2,Sig2,Vh2 = np.linalg.svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = np.diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.inv(Sig)) # build A tilde
    mu,W = np.linalg.eig(Atil)
    Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W) # build DMD modes1
    return mu, Phi

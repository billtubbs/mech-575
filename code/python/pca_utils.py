# Basic Python equivalent to MATLAB pca function
import numpy as np

def pca(X):

    # Number of datapoints
    n = X.shape[0]
    
    # Compute the row-wise mean 
    Xavg = X.mean(axis=0)
    
    # Compute the mean-subtracted data
    B = X - Xavg
    
    # Compute SVD
    U, S, VH = np.linalg.svd(B.T/np.sqrt(n), full_matrices=False)
    
    import pdb; pdb.set_trace()
    
    coeff = U
    score = None
    latent = None
    
    return coeff, score, latent

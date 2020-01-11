from math import sin, cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

theta = pi/3
xC = np.array([2, 1]).reshape(-1, 1)
sig = np.array([2, .5])
R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

# Generate random data
n = 10000
X = np.dot(R*sig, np.random.randn(2, n)) \
    + np.repeat(xC, n, axis=1)

# Compute PCA via SVD
Xavg = X.mean(axis=1)
B = X - Xavg.reshape(-1, 1)
U, S, VH = np.linalg.svd(B/sqrt(n), full_matrices=False)

# How to compute using PCA
#pca = PCA(n_components=2)
#pca.fit(X)

# Compute confidence intervals
theta = np.linspace(0, 1, 100) * 2 * pi
circle_pts = np.stack([np.cos(theta), np.sin(theta)])
Xstd = np.dot(U*S, circle_pts)

# Plot figure
plt.scatter(X[0,:], X[1,:], marker='.')
plt.plot(Xavg[0] + Xstd[0, :], Xavg[1] + Xstd[1, :], color='r')
plt.plot(Xavg[0] + 2*Xstd[0, :], Xavg[1] + 2*Xstd[1, :], color='r')
plt.plot(Xavg[0] + 3*Xstd[0, :], Xavg[1] + 3*Xstd[1, :], color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

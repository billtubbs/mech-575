from math import sin, cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

# Initialize random number generator
rng = np.random.RandomState(1)

theta = pi/3
xC = np.array([2, 1]).reshape(-1, 1)
sig = np.array([2, .5])
R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

# Generate random data
n = 10000
X = np.dot(R*sig, rng.randn(2, n)) + xC

# Compute PCA via SVD
Xavg = X.mean(axis=1)
B = X - Xavg.reshape(-1, 1)
U, S, VH = np.linalg.svd(B/sqrt(n), full_matrices=False)

# Data for Table 1.1
print("Standard deviation of data and normalized singular values.")
print({
    'Data': sig.tolist(),
    'SVD': S.tolist()
})
assert np.allclose(S, sig, atol=0.01)

# Compute confidence intervals
theta = np.linspace(0, 1, 100) * 2 * pi
circle_pts = np.stack([np.cos(theta), np.sin(theta)])
Xstd = np.dot(U*S, circle_pts)

# Plot figure
plt.figure(figsize=(5, 5))
plt.scatter(X[0,:], X[1,:], marker='.', alpha=0.2)
plt.plot(Xavg[0] + Xstd[0, :], Xavg[1] + Xstd[1, :], color='r')
plt.plot(Xavg[0] + 2*Xstd[0, :], Xavg[1] + 2*Xstd[1, :], color='r')
plt.plot(Xavg[0] + 3*Xstd[0, :], Xavg[1] + 3*Xstd[1, :], color='r')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.tight_layout()
plt.savefig('../../plots/fig_1_13b_py.pdf')
plt.savefig('../../plots/fig_1_13b_py.png')
plt.show()

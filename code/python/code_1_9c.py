from math import sin, cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initialize random number generator
rng = np.random.RandomState(1)

theta = pi/3
xC = np.array([2, 1]).reshape(-1, 1)
sig = np.array([2, .5])
R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

# Generate random data
n = 10000
X = (np.dot(R*sig, rng.randn(2, n)) + xC).T

# Compute PCA using sklearn
pca = PCA(n_components=2)
pca.fit(X)
components = pca.components_
explained_variance = pca.explained_variance_

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.2)
for length, vector in zip(explained_variance, components):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.tight_layout()
plt.savefig('../../plots/fig_1_13c_py.pdf')
plt.savefig('../../plots/fig_1_13c_py.png')
plt.show()

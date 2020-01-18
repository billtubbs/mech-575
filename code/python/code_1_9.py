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

# Plot figure
plt.figure(figsize=(5, 5))
plt.scatter(X[0,:], X[1,:], marker='.', alpha=0.2)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.tight_layout()
plt.savefig('../../plots/fig_1_13a_py.pdf')
plt.savefig('../../plots/fig_1_13a_py.png')
plt.show()

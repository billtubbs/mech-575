import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
x_mean = np.array([2, 1])
sigma = np.array([2, 5])

# Generate random data
theta = np.pi/3
R = np.array([[np.cos(theta), -np.sin(theta)],  
              [np.sin(theta), np.cos(theta)]])   
n = 10000
X = R.dot(np.diag(sigma)).dot(np.random.randn(2,n)) \
          + np.diag(x_mean).dot(np.ones((2,n)))
plt.scatter(X[0,:], X[1,:], marker='.')
plt.grid()
plt.show()

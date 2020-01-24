import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

A = np.array([[1, 2], [4, 3]])
D, T = np.linalg.eig(A)
# These work also:
#T = [-1 0.5; 1 1];
#D = [-1 0; 0 5];

n = 8
t_n = 0.4
x0 = np.array([[1], [1]])
T_inv = np.linalg.inv(T)
t_range = np.linspace(0, t_n, n+1)
x = [T.dot(expm(np.diag(D)*t)).dot(T_inv).dot(x0) for t in t_range]
x = np.hstack(x)

plt.plot(t_range, x[0], 'o-', label='$x_1$')
plt.plot(t_range, x[1], 'o-', label='$x_2$')
plt.legend()
plt.xlabel('t')
plt.ylabel('x')
plt.grid()
plt.show()

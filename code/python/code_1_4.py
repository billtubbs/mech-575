import numpy as np
import matplotlib.pyplot as plt


# Initialize random number generator
rng = np.random.RandomState(1)


x = 3  # True slope
a = np.arange(-2, 2, 0.25).reshape(-1, 1)

# Add noise
b = a*x + rng.randn(a.size).reshape(-1, 1)

plt.figure(figsize=(4, 4))

plt.plot(a, x*a, color='k', label='True line')
plt.scatter(a, b, marker='+', color='r', label='noisy measurements')

plt.xlabel('$a$')
plt.ylabel('$b$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('../../plots/fig_1_9_py.pdf')
plt.savefig('../../plots/fig_1_9_py.png')
plt.show()

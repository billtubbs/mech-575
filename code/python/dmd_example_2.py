# DMD Demo on Video Data
# 
# Author: Bill Tubbs
# Date: January 29, 2020.
# 
# Run these two scripts in sequence:
#  1. dmd_example_1.py - downloads video file and computes DMD
#  2. dmd_example_2.py - Simulate the dynamic system
# 
# This script loads the dynamic mode decomposition results from the data
# file and plots the eigenvalues.  TODO: Simulate the system


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from dyn_utils import DMD_SB


data_dir = '../../data'
plot_dir = '../../plots'
data_filename = 'video_data.npy'
dmd_filename_format = 'dmd_sol_{:d}.npz'

print(f"Loading video data file...")
filepath = os.path.join(data_dir, data_filename)
frame_data = np.load(filepath)
frame_data = frame_data.reshape(297, -1).T

print(f"Full data set: {frame_data.shape}")

# Data dimensions
n, m = frame_data.shape

x = frame_data[:, :-1]
x_prime = frame_data[:, 1:]

print("Data matrices")
print(f"x: {x.shape}")
print(f"x_prime: {x_prime.shape}")

# Set truncation parameter
r = 25
print(f"r: {r}")

# Compute or load DMD solution
dmd_filename = dmd_filename_format.format(r)
filepath = os.path.join(data_dir, dmd_filename)
if not os.path.exists(filepath):
    print(f"Computing DMD...")
    t0 = time.time()
    phi, lam, b = DMD_SB(x, x_prime, r)
    run_time = time.time() - t0
    print(f"Runtime: {run_time:.1f} s")
    # Check lam is diagonal
    assert np.count_nonzero(lam - np.diag(np.diagonal(lam))) == 0
    # Only save the diagonal values of lam
    np.savez(filepath, phi=phi, lam=np.diagonal(lam), b=b)
else:
    # Load previous solution from file
    data = np.load(filepath)
    phi = data['phi']
    lam = np.diag(data['lam'])
    b = data['b']
    print(f"DMD solution loaded from file.")

print(f"phi: {phi.shape}")
print(f"lam: {lam.shape}")
print(f"b: {b.shape}")

theta = np.linspace(0, 2*np.pi, 360)
circle_pts = (np.sin(theta), np.cos(theta))

# Plot eigenvalues
plt.figure(figsize=(4, 4))
plt.plot(circle_pts[0], circle_pts[1], linewidth=0.5, label='Unit circle')
plt.scatter(lam.real, lam.imag, marker='.', label='Eigenvalues')
plt.title(f'Eigenvalues ($r = {r}$)')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.xlabel(r'real')
plt.ylabel('imag')
plt.grid()
plt.tight_layout()
filepath = os.path.join(plot_dir, "dyn_example2_eig_plot.pdf")
plt.savefig(filepath)
plt.show()

#breakpoint()


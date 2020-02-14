import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_plot_3d(data, labels=('x', 'y', 'z'), 
                    show=True, cmap='RdYlBu', 
                    figsize=(9, 7), filename=None):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    cm = plt.cm.get_cmap(cmap)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    t = range(len(data))
    cbar = ax.scatter(x, y, z, c=t, marker='.', cmap=cm)
    cbar = plt.colorbar(cbar)
    cbar.set_label('Time (t)')

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)

    if show:
        plt.show()
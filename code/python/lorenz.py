"""Python functions for simulating the Lorenz System.
"""

def lorenz_odes(t, y, sigma, beta, rho):
    """The Lorenz system of ordinary differential equations.
    
    Returns:
        dydt (tuple): Derivative (w.r.t. time)
    """
    y1, y2, y3 = y
    return (sigma * (y2 - y1), y1 * (rho - y3) - y2, y1 * y2 - beta * y3)


def lorenz_odes_vectorized(t, y, sigma, beta, rho):
    """The Lorenz system of ordinary differential equations.

    Returns:
        dydt (np.array): Derivatives (w.r.t. time)
    """
    dydt = np.empty_like(y)
    dydt[0, ...] = sigma * (y[1] - y[0])
    dydt[1, ...] = y[0] * (rho - y[2]) - y[1]
    dydt[2, ...] = y[0] * y[1] - beta * y[2]
    
    return dydt


def scatter_plot_3d(data, show=True, cmap='RdYlBu', 
                    labels=('x', 'y', 'z'),
                    figsize=(9, 7)):
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

    if show:
        plt.show()

"""Python functions for simulating Enzyme Reaction Kinetics.

This example is the example presented by Niall Mangan in the paper
"Inferring biological networks by sparse identification of 
nonlinear dynamics" by N. M. Mangan, S. L. Brunton, J. L. Proctor,
and J. N. Kutz.
"""

import numpy as np
import matplotlib.pyplot as plt


def mmkinetics(t, x, v_max=1.5, k_m=0.3, j_in=0.6):
    """The Michaelis-Menten kinetics model of enzymatic reactions.

    Args:
        t (float or array): Time value (not used)
        x (float or array): State variable (concentration)
        v_max (float): Maximum reaction rate
        k_m (float): Half max reaction rate
        j_in (float): Influx of substrate
    
    Returns:
        dxdt (float): Derivative of x (w.r.t. time)
    """
    return j_in - v_max * x / (k_m + x)



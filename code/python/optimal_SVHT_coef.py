from math import pi
import numpy as np
import scipy.integrate as integrate


def optimal_SVHT_coef(beta, sigma_known):
    """Coefficient determining optimal location of Hard Threshold for Matrix
    Denoising by Singular Values Hard Thresholding when noise level is known or
    unknown.

    See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
    Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870

    Args:
        beta (float or vector): aspect ratio m/n of the matrix to be
            denoised, 0 < beta <= 1.
        sigma_known (float): 1 if noise level known, 0 if unknown.
    
    Returns:
        coef (float): optimal location of hard threshold, up the median data singular
            value (sigma unknown) or up to sigma*sqrt(n) (sigma known);
            a vector of the same dimension as beta, where coef[i] is the
            coefficient corresponding to beta[i].

    Usage in known noise level:

    Given an m-by-n matrix Y known to be low rank and observed in white noise
    with mean zero and known variance sigma^2, form a denoised matrix Xhat by:

    >>> U, D, V = np.linalg.svd(Y)
    >>> y = np.diag(Y)
    >>> y[y < (optimal_SVHT_coef(m/n, 1) * sqrt(n) * sigma)] = 0
    >>> Xhat = U * diag(y) * V.T

    Usage in unknown noise level:

    Given an m-by-n matrix Y known to be low rank and observed in white
    noise with mean zero and unknown variance, form a denoised matrix
    Xhat by:
     
    >>> U, D, V = np.linalg.svd(Y)
    >>> y = diag(Y)
    >>> y[y < (optimal_SVHT_coef_sigma_unknown(m/n, 0) * median(y))] = 0
    >>> Xhat = U * np.diag(y) * V.T
    
    -----------------------------------------------------------------------------
    Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013
    
    This program is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    this program.  If not, see <http://www.gnu.org/licenses/>.
    -----------------------------------------------------------------------------
    """

    if sigma_known:
        return optimal_SVHT_coef_sigma_known(beta)
    else:
        return optimal_SVHT_coef_sigma_unknown(beta)


def optimal_SVHT_coef_sigma_known(beta):
    assert(all(beta > 0))
    assert(all(beta <= 1))
    assert(prod(size(beta)) == len(beta))  # beta must be a vector
    w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
    lambda_star = np.sqrt(2 * (beta + 1) + w)
    return lambda_star


def optimal_SVHT_coef_sigma_unknown(beta):
    warning('off', 'MATLAB:quadl:MinStepSize')
    assert(all(beta>0))
    assert(all(beta<=1))
    assert(prod(size(beta)) == len(beta))  # beta must be a vector
    coef = optimal_SVHT_coef_sigma_known(beta)
    MPmedian = np.zeros_like(beta)
    for i in range(beta):
        MPmedian[i] = median_marcenko_pastur(beta[i])
    omega = coef / np.sqrt(MPmedian)
    return omega


def MarcenkoPasturIntegral(x, beta):
    if beta <= 0 or beta > 1:
        raise ValueError('beta beyond')
    lobnd = (1 - np.sqrt(beta))^2
    hibnd = (1 + np.sqrt(beta))^2
    if (x < lobnd) or (x > hibnd):
        raise ValueError('x beyond')
    dens = lambda t: np.sqrt((hibnd-t) * (t-lobnd)) / (2*pi * beta * t)
    I = integrate(dens, lobnd, x)
    printf('x={x:.3f},beta={beta:.3f},I={I:.3f}\n')
    return I


def median_marcenko_pastur(beta):
    MarPas = lambda x: 1 - incMarPas(x, beta, 0)
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    change = 1
    while change & (hibnd - lobnd > .001):
      change = 0
      x = np.linspace(lobnd, hibnd, 5)
      for i in range(x):
          y[i] = MarPas(x[i])
      if any(y < 0.5):
         lobnd = max(x(y < 0.5))
         change = 1
      if any(y > 0.5):
         hibnd = min(x(y > 0.5))
         change = 1
    med = (hibnd+lobnd) / 2
    return med


def incMarPas(x0,beta,gamma):
    if beta > 1:
        raise ValueError('betaBeyond')
    topSpec = (1 + np.sqrt(beta))**2
    botSpec = (1 - np.sqrt(beta))**2
    MarPas = lambda x: IfElse((topSpec-x).*(x-botSpec) >0, ...
                         sqrt((topSpec-x).*(x-botSpec))./(beta.* x)./(2 .* pi), ...
                         0)
    if gamma ~= 0:
       fun = lambda x: (x**gamma * MarPas(x))
    else:
       fun = lambda x: MarPas(x)
    end
    I = integrate(fun, x0, topSpec)
    function y=IfElse(Q, point, counterPoint)
        y = point;
        if any(~Q):
            if len(counterPoint) == 1:
                counterPoint = np.ones_like(Q) * counterPoint
            y(~Q) = counterPoint(~Q)
    return I
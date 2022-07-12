"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Contains functions related to the Lotka-Volterra system of ODEs.

Contents:
1) type_II_lotka_volterra(P, t = 0)
  The type-II lotka-volterra ode system.

2) type_II_equilibrium(params)
  The stable equilibrium point for a given set of ODE parameters.
"""

import numpy as np

# The ODE system of equations
def type_II_lotka_volterra(P, t, a, b, c, d, h, K, m, z = 0):
    """
    Parameters:
    -----------
    P: array [1x2]
        The populations of prey and predator; P[0] is prey and P[1] is predator.
    params: list
        A list of the 6 model parameters for the type-II equation;
        [a, b, c, d, h, K]

    Returns:
    --------
    """
    P = [b*P[0]*(1-P[0]/K) - (a*P[0]*P[1])/(P[1]**m + a*h*P[0]) - z*P[0],
    c*(a*P[0]*P[1])/(P[1]**m + a*h*P[0]) - d*P[1]]
    return P

# The equilibrium condition
def type_II_equilibrium(params):
    """
    Parameters:
    -----------
    params: list
    A list of the 6 model parameters for the type-II equation;
    [a, b, c, d, h, K]

    Returns:
    --------
    EQ : array [1x2]
        The equilibrium point for the ODE system.
        E[0] is the equilibrium population for the prey,
        E[1] is the equilibrium population for the predator.
    """

    # Re-define specific parameters
    a, b, c, d, h, K, m, z = [x for x in params]

    EQ = [(d)/(a*(c-d*h)), b*(1+a*h*(d/(a*(c-d*h))))*(1-(d/(a*(c-d*h)))/K)/a]
    return EQ

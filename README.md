# Harvested Predator-Prey System - Tutorial Series
Contains scripts used to explore the dynamics of a harvested predator prey system.

This tutorial series is inspired by the work of Hadjimichael, Reed, and Quinn (2020), [_Navigating Deeply Uncertain Tradeoffs in a Harvested Predator-Prey System_](https://www.hindawi.com/journals/complexity/2020/4170453/).

A series of blog posts have been written to guide the user through the tutorial, and can be accessed at [https://waterprogramming.wordpress.com/](https://waterprogramming.wordpress.com/).

## Overview of tutorial series
### Post 0: Introduction to Predator-Prey System Dynamics

The first post in the tutorial series, [accessible at this link](https://waterprogramming.wordpress.com/2022/07/11/__trashed-3/) focuses on the ordinary differential equations (ODEs) which define the predator-prey system.
A predator-dependent variation of the classic Lotka-Volterra equations is used, predator interference and harvesting are also considered.

Stability conditions are derived, and an interactive Jupyter Notebook widget is used to explore the various stability conditions resulting from different parameter value combinations (.gif demonstration shown below).

<p align="center">
    <img src="https://github.com/TrevorJA/harvested_predator_prey_system_tutorial_series/blob/main/Part_0_ODE_Dynamics/example_figures/Animation_stable.gif" alt = "Demonstration of the interactive ODE widget." />
</p>


### Post 1: Harvest Optimization and MOEA Diagnostics

The second post in the series, [accessible here through the WaterProgramming blog](https://waterprogramming.wordpress.com/2022/08/08/fisheries-training-part-1-harvest-optimization-and-moea-diagnostics/), or via the [Jupyter Notebook in the repository](https://github.com/TrevorJA/harvested_predator_prey_system_tutorial_series/blob/main/Part_1_MOEA_Diagnostics/Part%201%20-%20Harvest%20Optimization%20and%20MOEA%20Diagnostics.ipynb), studies the formulation of harvesting strategies via radial basis functions, and the multi-objective optimization of harvesting strategies within the predator-prey system.

Radial basis functions are described as being flexible and useful for generating adaptive harvesting policies.

<p align="center">
    <img src="https://github.com/TrevorJA/harvested_predator_prey_system_tutorial_series/blob/main/Part_1_MOEA_Diagnostics/many_random_RBFs.png" alt = "Demonstration of the flexibility of radial basis functons (RBFs)." />
</p>


Multiple harvesting objectives are defined, including:
1. Discounted profits
2. Prey population deficit
3. Longest duration of consecutive law
4. Worst harvest instance
5. Harvest variance

Additionally, a constraint is included which _avoids collapse of predator population_.

The [Borg Multi-objective evolutionary algorithm](http://borgmoea.org/) (MOEA) is used to optimize a state-aware adaptive harvesting policy, which prescribes harvesting efforts dependent upon the current prey population levels. The Borg MOEA is employed through a python-wrapper (pyBorg).

Basic MOEA diagnostics are performed including an analysis of hypervolume growth with respect to the number of function evaluations, and a random seed analysis which shows the insensitivity of the PyBorg MOEA to initial algorithm parameterizations.

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59
Code adapted from https://waterprogramming.wordpress.com/2021/08/13/introduction-to-pyborg-basic-setup-and-running/
"""

# import all required libraries
from platypus import (Problem, Real, Hypervolume, Generator)
from pyborg import BorgMOEA
from runtime_diagnostics import runtime_hvol
from fish_game import fish_game, plot_3d_tradeoff, plot_runtime
import matplotlib.pyplot as plt
import time

#%%
"""
Code to run BorgMOEA starts here
Obj1: Mean NPV for all realizations
Obj2: Mean prey deficit
Obj3: Mean worst case of consecutive low harvest across realizations
Obj4: Mean 1st percentile of all harvests
Obj5: Mean variance of harvest

"""

# Based on Hadjimichael et al 2020
nVars = 6   # Define number of decision variables
nObjs = 5   # Define number of objective -- USER DEFINED
#nObjs = 3 # change to 3 to make the problem easier
nCnstr = 1      # Define number of decision constraints

problem = Problem(nVars, nObjs, nCnstr)

# set bounds for each decision variable
problem.types[0] = Real(0.0, 1.0)
problem.types[1] = Real(0.0, 1.0)
problem.types[2] = Real(0.0, 1.0)
problem.types[3] = Real(0.0, 1.0)
problem.types[4] = Real(0.0, 1.0)
problem.types[5] = Real(0.0, 1.0)


# all values should be nonzero
problem.constraints[:] = "==0"

# set problem function
problem.function = fish_game

algorithm = BorgMOEA(problem, epsilons=0.001)

# begin timing the borg run
borg_start_time = time.time()
algorithm.run(100)
borg_end_time = time.time()

borg_total_time = borg_end_time - borg_start_time

print(f"borg_total_time={borg_total_time}s")

#%%

# Plot objective tradeoff surface
fig_objs = plt.figure()
ax_objs = fig_objs.add_subplot(111, projection='3d')

objs_indices = [0, 1, 2]
obj_labels = ['Mean NPV', 'Mean prey deficit', 'Mean WCLH']
obj_min = [-6000, 0, 0]

plot_3d_tradeoff(algorithm, ax_objs, objs_indices, obj_labels, obj_min)

#%%
# Plot hypervolume
# define detailed_run parameters
maxevals = 500
frequency = 100
output = "fishery.data"

# set inputs for measuring hypervolume
hv = Hypervolume(minimum=[-6000, 0, 0, -250, 0], maximum=[0, 1, 100, 0, 32000])

# Note: Cannot plot epsilon indicator and GD as those require reference sets, which
# the fisheries problem does not have

nfe, hyp = runtime_hvol(algorithm, maxevals, frequency, output, hv)

# plot hypervolume
#plot_runtime(nfe, hyp, 'PyBorg Runtime (Hypervolume)', 'Hypervolume')
plt.plot(nfe, hyp)
plt.title('PyBorg Runtime (Hypervolume)')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Hypervolume')
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59
Code adapted from https://waterprogramming.wordpress.com/2021/08/13/introduction-to-pyborg-basic-setup-and-running/
"""

# import all required libraries
from platypus import (Problem, Real, Hypervolume)
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
nVars = 9   # Define number of decision variables
nObjs = 5   # Define number of objective -- USER DEFINED
#nCnstr = 1      # Define number of decision constraints

problem = Problem(nVars, nObjs)     

# set bounds for each decision variable
problem.types[0] = Real(0.002,2.001)
problem.types[1] = Real(0.005,1.001)
problem.types[2] = Real(0.2,1.001)
problem.types[3] = Real(0.05,0.201)
problem.types[4] = Real(0.001,2.001)
problem.types[5] = Real(100,5000.001)
problem.types[6] = Real(0.1,1.501)
problem.types[7] = Real(0.001,0.011)
problem.types[8] = Real(0.001,0.011)

# all values should be nonzero
problem.constraints[:] = ">=0"

# set problem function
problem.function = fish_game

algorithm = BorgMOEA(problem, epsilons=0.001)

# begin timing the borg run
borg_start_time = time.time()
algorithm.run(1000)
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
maxevals = 4000
frequency = 100
output = "fishery.data"

# set inputs for measuring hypervolume
hv = Hypervolume(minimum=[-6000, 0, 0, 0, -32000], maximum=[0, 1, 100, 250, 0])

# Note: Cannot plot epsilon indicator and GD as those require reference sets, which 
# the fisheries problem does not have

nfe, hyp = runtime_hvol(algorithm, maxevals, frequency, output, hv)
print(f"hv = {hv}")

# plot hypervolume
#plot_runtime(nfe, hyp, 'PyBorg Runtime (Hypervolume)', 'Hypervolume')
plt.plot(nfe, hyp)
plt.title('PyBorg Runtime (Hypervolume)')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Hypervolume')
plt.show()


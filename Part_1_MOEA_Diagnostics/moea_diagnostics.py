# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59
Code adapted from https://waterprogramming.wordpress.com/2021/08/13/introduction-to-pyborg-basic-setup-and-running/
"""

# import all required libraries
from platypus import Problem, Real, Hypervolume, EpsilonIndicator, GenerationalDistance
from pyborg import BorgMOEA
from runtime_diagnostics import runtime_hvol, runtime_epsilon, runtime_gdistance
from fish_game import fish_game, plot_3d_tradeoff, plot_runtime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Code to run BorgMOEA starts here
Obj1: Mean NPV for all realizations
Obj2: Mean prey deficit
Obj3: Mean worst case of consecutive low harvest across realizations
Obj4: Mean 1st percentile of all harvests
Obj5: Mean variance of harvest

"""

# USER-DEFINED
# Define number of decision variables, objectives, and constraints
nVars = 9
nObjs = 5
nCnstr = 1 

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
algorithm.run(10000)


# Plotting begins here
# define detailed_run parameters
maxevals = 10000
frequency = 1000
output = "fishery.data"

# set inputs for measuring hypervolume
hv = Hypervolume(minimum=[-6000, 0, 0, 0, -32000], maximum=[0, 1, 100, 250, 0])
# set inputs for measuring epsilon indicator
epi = EpsilonIndicator(minimum=[-6000, 0, 0, 0, -32000], maximum=[0, 1, 100, 250, 0])
# set inputs for measuring generational distance
gd = GenerationalDistance(minimum=[-6000, 0, 0, 0, -32000], maximum=[0, 1, 100, 250, 0])

nfe, hyp = runtime_hvol(algorithm, maxevals, frequency, output, hv)
nfe, epsInd = runtime_epsilon(algorithm, maxevals, frequency, output, epi)
nfe, genDist = runtime_gdistance(algorithm, maxevals, frequency, output, gd)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

objs_indices = [0, 1, 2]
obj_labels = ['Mean NPV', 'Mean prey deficit', 'Mean WCLH']
obj_min = [-6000, 0, 0]

plot_3d_tradeoff(algorithm, ax, objs_indices, obj_labels, obj_min)

# plot hypervolume
plot_runtime(nfe, hyp, 'PyBorg Runtime Hypervolume Fish game', 'Hypervolume')
# plot epsilon indicator
plot_runtime(nfe, hyp, 'PyBorg Runtime Eps. Indicator Fish game', 'Epsilon indicator')
# plot generational distance
plot_runtime(nfe, hyp, 'PyBorg Runtime Gen. Distance Fish game', 'Gen. distance')



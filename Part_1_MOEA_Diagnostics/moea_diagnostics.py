# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59
Code adapted from https://waterprogramming.wordpress.com/2021/08/13/introduction-to-pyborg-basic-setup-and-running/
"""

# import all required libraries
import pyborg
from platypus import (Problem, Real, Hypervolume, Generator)
from pyborg import BorgMOEA
from fish_game_functions import (fish_game_5_objs, fish_game_3_objs, plot_3d_tradeoff, 
                                 runtime_hvol, plot_runtime, select_objective)
import matplotlib.pyplot as plt
import time
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Label

#%%
"""
Code to run BorgMOEA starts here
Obj1: Mean NPV for all realizations
Obj2: Mean prey deficit
Obj3: Mean worst case of consecutive low harvest across realizations
Obj4: Mean 1st percentile of all harvests
Obj5: Mean variance of harvest
"""
#%%
# Based on Hadjimichael et al 2020
<<<<<<< HEAD
prob_formulation = widgets.Dropdown(options=['5-objective', '3-objective'],
                         description='', value='5-objective', disabled=False)
display(prob_formulation)

#%%

nVars = 6   # Define number of decision variables
nObjs = 5
if prob_formulation == '5-objective':
    nObjs = 5
elif prob_formulation == '3-objective':
    nObjs = 3
nCnstr = 1      # Define number of decision constraints

#%%
problem = Problem(nVars, nObjs, nCnstr)     
=======
nVars = 6   # Define number of decision variables
nObjs = 5   # Define number of objective -- USER DEFINED
#nObjs = 3 # change to 3 to make the problem easier
nCnstr = 1      # Define number of decision constraints

problem = Problem(nVars, nObjs, nCnstr)
>>>>>>> 4104c6e99b625aab32cfa52558108d9a31af0a1d

# set bounds for each decision variable
problem.types[0] = Real(0.0, 1.0)
problem.types[1] = Real(0.0, 1.0)
problem.types[2] = Real(0.0, 1.0)
problem.types[3] = Real(0.0, 1.0)
problem.types[4] = Real(0.0, 1.0)
problem.types[5] = Real(0.0, 1.0)
<<<<<<< HEAD
=======

>>>>>>> 4104c6e99b625aab32cfa52558108d9a31af0a1d

# all values should be nonzero
problem.constraints[:] = "==0"

# set problem function
if nObjs == 5:
    problem.function = fish_game_5_objs
else:
    problem.function = fish_game_3_objs

algorithm = BorgMOEA(problem, epsilons=0.001)

# begin timing the borg run
borg_start_time = time.time()
algorithm.run(100)
borg_end_time = time.time()

borg_total_time = borg_end_time - borg_start_time

print(f"borg_total_time={borg_total_time}s")

#%%
# Interactively select objectives to plot
objs_list = ['Objective 1', 'Objective 2', 'Objective 3']

obj1 = widgets.Dropdown(options=['Mean NPV', 'Mean prey deficit', 'Mean WCLH', 
                                 'Mean 1% harvest', 'Mean harvest variance'],
                         description='', value='Mean NPV', disabled=False)
obj2 = widgets.Dropdown(options=['Mean NPV', 'Mean prey deficit', 'Mean WCLH', 
                                 'Mean 1% harvest', 'Mean harvest variance'],
                         description='', value='Mean prey deficit', disabled=False)

obj3 = widgets.Dropdown(options=['Mean NPV', 'Mean prey deficit', 'Mean WCLH', 
                                 'Mean 1% harvest', 'Mean harvest variance'],
                         description='', value='Mean WCLH', disabled=False)

list_of_objs = [Label(i) for i in objs_list]
list_of_dropdowns = [obj1, obj2, obj3]

HBox([VBox(objs_list), VBox(list_of_dropdowns)])
box_layout = widgets.Layout(display='flex', flex_flow = 'row', align_items ='center', justify_content = 'center')

#%%
# Plot objective tradeoff surface
fig_objs = plt.figure()
ax_objs = fig_objs.add_subplot(111, projection='3d')

objs_indices = []
objs_labels = []
objs_min = []

if nObjs == 5:
    objs_indices = [select_objective(obj1)[0], select_objective(obj2)[0], select_objective(obj2)[0]]
    obj_labels = [obj1, obj2, obj3]
    obj_min = [select_objective(obj1)[1], select_objective(obj2)[1], select_objective(obj2)[1]]
else:
    objs_indices = [0,1,2]
    obj_labels = ['Mean NPV', 'Mean prey deficit', 'Mean WCLH']
    obj_min = [-6000, 0, 0]

plot_3d_tradeoff(algorithm, ax_objs, objs_indices, obj_labels, obj_min)

#%%
# Plot hypervolume
# define detailed_run parameters
maxevals = 5000
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

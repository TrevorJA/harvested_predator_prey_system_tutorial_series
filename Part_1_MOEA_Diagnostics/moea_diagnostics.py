# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59
Code adapted from https://waterprogramming.wordpress.com/2021/08/13/introduction-to-pyborg-basic-setup-and-running/
"""

# import all required libraries
from platypus import (Problem, Real, Hypervolume)
from pyborg import BorgMOEA
from fish_game_functions import *
import matplotlib.pyplot as plt
import time
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Label
import random

#%%
"""
Obj1: Mean NPV for all realizations
Obj2: Mean prey deficit
Obj3: Mean worst case of consecutive low harvest across realizations
Obj4: Mean 1st percentile of all harvests
Obj5: Mean variance of harvest
"""

#%%
# Based on Hadjimichael et al 2020
prob_formulation = widgets.Dropdown(options=['5-objective', '3-objective'],
                         description='', value='5-objective', disabled=False)
display(prob_formulation)

#%%

# Set the number of decision variables, constraints and performance objectives
nVars = 6   # Define number of decision variables
nObjs = 5
if prob_formulation == '5-objective':
    nObjs = 5
elif prob_formulation == '3-objective':
    nObjs = 3
nCnstr = 1      # Define number of decision constraints

# Define the upper and lower bounds of the performance objectives
objs_lower_bounds = [-6000, 0, 0, -250, 0]
objs_upper_bounds = [0, 1, 100, 0, 32000]


#%%

# initialize the optimization
init_list = ['Initial NFE: ', 'Initial population size: ']
init_labels = [Label(i) for i in init_list]

init_nfe = widgets.IntText(value=100, description='Any', disabled=False)
init_pop_size = widgets.IntText(value=100, description='Any', disabled=False)

list_of_inits = [init_nfe, init_pop_size]

HBox([VBox(init_labels), VBox(list_of_inits)])
box_layout_inits = widgets.Layout(display='flex', flex_flow = 'row', align_items ='center', justify_content = 'center')

#%%

# begin timing the Borg run
borg_start_time = time.time()

algorithm = fisheries_game_problem_setup(nVars, nObjs, nCnstr, pop_size=int(init_pop_size))
algorithm.run(int(init_nfe))

# end timing and print optimization time 
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

HBox([VBox(list_of_objs), VBox(list_of_dropdowns)])
box_layout_objs = widgets.Layout(display='flex', flex_flow = 'row', align_items ='center', justify_content = 'center')

#%%
# Plot objective tradeoff surface
fig_objs = plt.figure(figsize=(8,8))
ax_objs = fig_objs.add_subplot(111, projection='3d')

objs_indices = []
objs_labels = []
objs_min = []

if nObjs == 5:
    objs_indices = [select_objective(obj1.value)[0], select_objective(obj2.value)[0], select_objective(obj2.value)[0]]
    obj_labels = [obj1.value, obj2.value, obj3.value]
    obj_min = [select_objective(obj1.value)[1], select_objective(obj2.value)[1], select_objective(obj2.value)[1]]
else:
    objs_indices = [0,1,2]
    obj_labels = ['Mean NPV', 'Mean prey deficit', 'Mean WCLH']
    obj_min = [-6000, 0, 0]

plot_3d_tradeoff(algorithm, ax_objs, objs_indices, obj_labels, obj_min)

#%%
# Plot hypervolume
# define detailed_run parameters


#%%
# begin plotting here
maxevals = widgets.IntText(value=int(init_nfe)*2, description='Any', disabled=False)
frequency = widgets.IntText(value=int(init_nfe), description='Any', disabled=False)

fig_hvol = plt.figure(figsize=(8,12))
ax_hvol = fig_hvol.add_subplot()


plot_hvol(algorithm, maxevals, frequency, objs_lower_bounds, objs_upper_bounds, ax_hvol)

plt.title('PyBorg Runtime (Hypervolume)')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Hypervolume')
plt.show()

#%% 
# Random seed analysis here
# Performed to show if pyBorg's performance is sensitive to the size of its initial population
pop_size_list = [100, 200, 400, 800, 1000]

fig_rand_seed = plt.figure(figsize=(8,12))
ax_rand_seed = fig_rand_seed.add_subplot()

for p in range(len(pop_size_list)):
    fisheries_game_problem_setup(nVars, nObjs, nCnstr, pop_size_list[p])
    algorithm = fisheries_game_problem_setup(nVars, nObjs, nCnstr, pop_size=int(init_pop_size))
    algorithm.run(int(init_nfe))
    
    plot_hvol(maxevals, frequency, objs_lower_bounds, objs_upper_bounds, 
              ax_rand_seed, pop_size=pop_size_list[p])

plt.title('PyBorg Random Seed Analysis')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Hypervolume')
plt.legend()
plt.show()
    

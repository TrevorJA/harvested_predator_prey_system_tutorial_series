# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59
Code adapted from https:\\waterprogramming.wordpress.com\2021\08\13\introduction-to-pyborg-basic-setup-and-running\
"""

# import all required libraries
from platypus import (Problem, Real, Hypervolume)
import itertools
import matplotlib.pyplot as plt
import time
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Label
import random
import numpy as np

#%%
"""
Obj1: Mean NPV for all realizations
Obj2: Mean prey deficit
Obj3: Mean worst case of consecutive low harvest across realizations
Obj4: Mean 1st percentile of all harvests
Obj5: Mean variance of harvest
"""

import sys, os
sys.path.append('pyborg')

def harvest_strategy(Inputs, vars, input_ranges, output_ranges, nIn, nOut, nRBF):
    """
    Calculate outputs (u) corresponding to each sample of inputs

    Parameters
    ----------
    Inputs : TYPE
        DESCRIPTION.
    vars : TYPE
        DESCRIPTION.
    input_ranges : TYPE
        DESCRIPTION.
    output_ranges : TYPE
        DESCRIPTION.

    Returns
    -------
    norm_u : matrix
        A 2D matrix with nOut columns and as many rows as there are samples of
        input.

    """
    # Rearrange decision variables into C, R, and W arrays
    # C and R are nIn x nRBF and W is nOut x nRBF
    # Decision variables are arranged in 'vars' as nRBF consecutive
    # sets of {nIn pairs of {C, R} followed by nOut Ws}
    # E.g. for nRBF = 2, nIn = 3 and nOut = 4:
    # C, R, C, R, C, R, W, W, W, W, C, R, C, R, C, R, W, W, W, W
    C = np.zeros([nIn,nRBF])
    R = np.zeros([nIn,nRBF])
    W = np.zeros([nOut,nRBF])

    for n in range(nRBF):
        for m in range(nIn):
            C[m,n] = vars[(2*nIn+nOut)*n + 2*m]
            R[m,n] = vars[(2*nIn+nOut)*n + 2*m + 1]
        for k in range(nOut):
            W[k,n] = vars[(2*nIn+nOut)*n + 2*nIn + k]

    # Normalize weights to sum to 1 across the RBFs (each row of W should sum to 1)
    totals = np.sum(W,1)
    for k in range(nOut):
        if totals[k] > 0:
            W[k,:] = W[k,:]/totals[k]

    # Normalize inputs
    norm_in = np.zeros(nIn)
    for m in range (nIn):
        norm_in[m] = (Inputs[m]-input_ranges[m][0])/(input_ranges[m][1]-input_ranges[m][0])
    # Create array to store outputs
    u = np.zeros(nOut)
    # Calculate RBFs
    for k in range(nOut):
        for n in range(nRBF):
            BF = 0
            for m in range(nIn):
                if R[m,n] > 10**-6: # set so as to avoid division by 0
                    BF = BF + ((norm_in[m]-C[m,n])/R[m,n])**2
                else:
                    BF = BF + ((norm_in[m]-C[m,n])/(10**-6))**2
            u[k] = u[k] + W[k,n]*np.exp(-BF)
    # De-normalize outputs
    norm_u = np.zeros(nOut)
    for k in range(nOut):
        norm_u[k] = output_ranges[k][0] + u[k]*(output_ranges[k][1]-output_ranges[k][0])
    return norm_u


def fish_game_5_objs(vars):
    """
    Defines the full, 5-objective fish game problem to be solved

    Parameters
    ----------
    vars : list of floats
        Contains the C, R, W values

    Returns objs, cnstr

    """

    # Get chosen strategy
    strategy = 'Previous_Prey'

    # Define variables for RBFs
    nIn = 1 # no. of inputs (depending on selected strategy)
    nOut = 1 # no. of outputs (depending on selected strategy)
    nRBF = 2 # no. of RBFs to use

    nObjs = 5
    nCnstr = 1 # no. of constraints in output

    tSteps = 100 # no. of timesteps to run the fish game on
    N = 100 # Number of realizations of environmental stochasticity

    # Get system behavior parameters (need to convert from string to float)
    a = 0.005
    b = 0.5
    c = 0.5
    d = 0.1
    h = 0.1
    K = 2000
    m = 0.7
    sigmaX = 0.004
    sigmaY = 0.004

    x = np.zeros(tSteps+1) # Create prey population array
    y = np.zeros(tSteps+1) # Create predator population array
    z = np.zeros(tSteps+1) # Create harvest array

    # Create array to store harvest for all realizations
    harvest = np.zeros([N,tSteps+1])
    # Create array to store effort for all realizations
    effort = np.zeros([N,tSteps+1])
    # Create array to store prey for all realizations
    prey = np.zeros([N,tSteps+1])
    # Create array to store predator for all realizations
    predator = np.zeros([N,tSteps+1])

    # Create array to store metrics per realization
    NPV = np.zeros(N)
    cons_low_harv = np.zeros(N)
    harv_1st_pc = np.zeros(N)
    variance = np.zeros(N)

    # Create arrays to store objectives and constraints
    objs = [0.0]*nObjs
    cnstr = [0.0]*nCnstr

    # Create array with environmental stochasticity for prey
    epsilon_prey = np.random.normal(0.0, sigmaX, N)

    # Create array with environmental stochasticity for predator
    epsilon_predator = np.random.normal(0.0, sigmaY, N)

    # Go through N possible realizations
    for i in range(N):
        # Initialize populations and values
        x[0] = prey[i,0] = K
        y[0] = predator[i,0] = 250
        z[0] = effort[i,0] = harvest_strategy([x[0]], vars, [[0, K]], [[0, 1]], nIn, nOut, nRBF)
        NPVharvest = harvest[i,0] = effort[i,0]*x[0]
        # Go through all timesteps for prey, predator, and harvest
        for t in range(tSteps):
            if x[t] > 0 and y[t] > 0:
                x[t+1] = (x[t] + b*x[t]*(1-x[t]/K) - (a*x[t]*y[t])/(np.power(y[t],m)+a*h*x[t]) - z[t]*x[t])* np.exp(epsilon_prey[i]) # Prey growth equation
                y[t+1] = (y[t] + c*a*x[t]*y[t]/(np.power(y[t],m)+a*h*x[t]) - d*y[t]) *np.exp(epsilon_predator[i]) # Predator growth equation
                if t <= tSteps-1:
                    if strategy == 'Previous_Prey':
                        input_ranges = [[0, K]] # Prey pop. range to use for normalization
                        output_ranges = [[0, 1]] # Range to de-normalize harvest to
                        z[t+1] = harvest_strategy([x[t]], vars, input_ranges, output_ranges, nIn, nOut, nRBF)
            prey[i,t+1] = x[t+1]
            predator[i,t+1] = y[t+1]
            effort[i,t+1] = z[t+1]
            harvest[i,t+1] = z[t+1]*x[t+1]
            NPVharvest = NPVharvest + harvest[i,t+1]*(1+0.05)**(-(t+1))
        NPV[i] = NPVharvest
        low_hrv = [harvest[i,j]<prey[i,j]/20 for j in range(len(harvest[i,:]))] # Returns a list of True values when there's harvest below 5%
        count = [ sum( 1 for _ in group ) for key, group in itertools.groupby( low_hrv ) if key ] # Counts groups of True values in a row
        if count: # Checks if theres at least one count (if not, np.max won't work on empty list)
            cons_low_harv[i] = np.max(count)  # Finds the largest number of consecutive low harvests
        else:
            cons_low_harv[i] = 0
        harv_1st_pc[i] = np.percentile(harvest[i,:],1)
        variance[i] = np.var(harvest[i,:])

    # Calculate objectives across N realizations
    objs[0] = -np.mean(NPV) # Mean NPV for all realizations
    objs[1] = np.mean((K-prey)/K) # Mean prey deficit
    objs[2] = np.mean(cons_low_harv) # Mean worst case of consecutive low harvest across realizations
    objs[3] = -np.mean(harv_1st_pc) # Mean 1st percentile of all harvests
    objs[4] = np.mean(variance) # Mean variance of harvest

    cnstr[0] = np.mean((predator < 1).sum(axis=1)) # Mean number of predator extinction days per realization

    # output should be all the objectives
    #return objs[0],objs[1],objs[2],objs[3],objs[4]
    return objs, cnstr


#%%
# Based on Hadjimichael et al 2020
prob_formulation = '5-objective'

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
init_nfe = 10000
init_pop_size = 100


#%%
#from fish_game_functions import fish_game_5_objs
from pyborg import BorgMOEA

algorithm = fish_game_5_objs(nVars, nObjs, nCnstr, pop_size=int(init_pop_size))

#open file to record data
output = "fishery_fullrun.data"
f = open(output, "w+")
f.write("# Variables = " + str(algorithm.problem.nvars))
f.write("\n# Objectives = " + str(algorithm.problem.nobjs) + "\n")

#%%
# begin timing the Borg run
borg_start_time = time.time()

algorithm.run(int(init_nfe))

# end timing and print optimization time 


#%%
# Run full run and record data
arch = algorithm.archive[:]
for i in range(len(arch)):
    sol = arch[i]
    for j in range(nvars):
        f.write(str(sol.variables[j]) + " ")
    for j in range(nobjs):
        f.write(str(sol.objectives[j]) + " ")
    f.write("\n")

borg_end_time = time.time()
borg_total_time = borg_end_time - borg_start_time
f.write("\nTime taken = " + str(borg_total_time))

f.close()

'''
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
    
'''
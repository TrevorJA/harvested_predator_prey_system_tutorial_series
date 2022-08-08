# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:38:41 2022

@author: lbl59

Code adapted from:
    Fish game: https://github.com/antonia-had/Generalized_fish_game/blob/master/generalized_fish_game.py

    Runtime diagnostics: Code adapted from Antonia Hadjimichael's original code
    Author: Antonia Hadjimichael (hadjimichael@psu.edu)

    Original code found in serial-borg-moea/Python/dtlz2_advanced.py
    Authors: Andrew Dircks & Dave Hadka
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import datetime
import pyborg
from platypus import (Problem, Real, Hypervolume)
from pyborg import BorgMOEA
from fish_game_functions import *
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Label
import random

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

    Returns
    -------
    objs, cnstr
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

    # Define assumed system parameters
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

            # Solve discretized form of ODE at subsequent time step
            if x[t] > 0 and y[t] > 0:
                x[t+1] = (x[t] + b*x[t]*(1-x[t]/K) - (a*x[t]*y[t])/(np.power(y[t],m)+a*h*x[t]) - z[t]*x[t])* np.exp(epsilon_prey[i]) # Prey growth equation
                y[t+1] = (y[t] + c*a*x[t]*y[t]/(np.power(y[t],m)+a*h*x[t]) - d*y[t]) *np.exp(epsilon_predator[i]) # Predator growth equation

                # Solve for harvesting effort at next timestep
                if t <= tSteps-1:
                    if strategy == 'Previous_Prey':
                        input_ranges = [[0, K]] # Prey pop. range to use for normalization
                        output_ranges = [[0, 1]] # Range to de-normalize harvest to
                        z[t+1] = harvest_strategy([x[t]], vars, input_ranges, output_ranges, nIn, nOut, nRBF)

            # Store values in arrays
            prey[i,t+1] = x[t+1]
            predator[i,t+1] = y[t+1]
            effort[i,t+1] = z[t+1]
            harvest[i,t+1] = z[t+1]*x[t+1]
            NPVharvest = NPVharvest + harvest[i,t+1]*(1+0.05)**(-(t+1))

        # Solve for objectives and constraint
        NPV[i] = NPVharvest
        low_hrv = [harvest[i,j]<prey[i,j]/20 for j in range(len(harvest[i,:]))] # Returns a list of True values when there's harvest below 5%
        count = [ sum( 1 for _ in group ) for key, group in itertools.groupby( low_hrv ) if key ] # Counts groups of True values in a row
        if count: # Checks if theres at least one count (if not, np.max won't work on empty list)
            cons_low_harv[i] = np.max(count)  # Finds the largest number of consecutive low harvests
        else:
            cons_low_harv[i] = 0
        harv_1st_pc[i] = np.percentile(harvest[i,:],1)
        variance[i] = np.var(harvest[i,:])

    # Average objectives across N realizations
    objs[0] = -np.mean(NPV) # Mean NPV for all realizations
    objs[1] = np.mean((K-prey)/K) # Mean prey deficit
    objs[2] = np.mean(cons_low_harv) # Mean worst case of consecutive low harvest across realizations
    objs[3] = -np.mean(harv_1st_pc) # Mean 1st percentile of all harvests
    objs[4] = np.mean(variance) # Mean variance of harvest

    cnstr[0] = np.mean((predator < 1).sum(axis=1)) # Mean number of predator extinction days per realization

    # output should be all the objectives, and constraint
    return objs, cnstr

def fish_game_3_objs(vars):
    objs, cnstr = fish_game_5_objs(vars)
    return objs[0:3], cnstr

def fisheries_game_problem_setup(nVars, nObjs, nCnstr, pop_size=100):
    """
    Sets up and runs the fisheries game for a given population size

    Parameters
    ----------
    nVars : int
        Number of decision variables.
    nObjs : int
        Number of performance objectives.
    nCnstr : int
        Number of constraints.
    pop_size : int, optional
        Initial population size of the randomly-generated set of solutions.
        The default is 100.

    Returns
    -------
    algorithm : pyBorg object
        The algorthm to optimize with a unique initial population size.

    """
    # Set up the problem
    problem = Problem(nVars, nObjs, nCnstr)
    nVars = 6   # Define number of decision variables
    nObjs = 5   # Define number of objective -- USER DEFINED
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
    if nObjs == 5:
        problem.function = fish_game_5_objs
    else:
        problem.function = fish_game_3_objs

    algorithm = BorgMOEA(problem, epsilons=0.001, population_size=pop_size)
    return algorithm


def plot_3d_tradeoff(algorithm, ax, nObjs, obj1, obj2, obj3):
    """
    Plots the 3D tradeoff space for three selected objectives.

    Parameters
    ----------
    algorithm: MOEA optimization object
        Algorithm object of which results are to be visualized
    ax : matplotlib object
        3D axis for plotting. Should already be initialized.
    obj_indices : list (int)
        List of objective indices to be plotted. Should be ordered such that the
        desired objectives are plotted as axes x, y, and z respectively.
    obj_labels : list (strings)
        List of objective labels to be plotted. Should be ordered such that the
        desired objectives are plotted as axes x, y, and z respectively.
    obj_min : list (int)
        List of minimum objective values to be plotted. Should be ordered such that the
        desired objectives are plotted as axes x, y, and z respectively.
    Returns
    -------
    None.

    """
    objs_indices = []
    objs_labels = []
    objs_min = []

    if nObjs == 5:
        objs_indices = [select_objective(obj1)[0],
                        select_objective(obj2)[0],
                        select_objective(obj3)[0]]
        objs_labels = [obj1, obj2, obj3]
        objs_min = [select_objective(obj1)[1],
                   select_objective(obj2)[1],
                   select_objective(obj2)[1]]
    else:
        objs_indices = [0,1,2]
        objs_labels = ['Mean NPV', 'Mean prey deficit', 'Mean WCLH']
        objs_min = [-6000, 0, 0]

    obj1_idx = objs_indices[0]
    obj2_idx = objs_indices[1]
    obj3_idx = objs_indices[2]

    obj1_lab = objs_labels[0]
    obj2_lab = objs_labels[1]
    obj3_lab = objs_labels[2]

    ax.scatter([s.objectives[obj1_idx] for s in algorithm.result],
               [s.objectives[obj2_idx] for s in algorithm.result],
               [s.objectives[obj3_idx] for s in algorithm.result],
               c='blue', s=100)

    ax.set_xlabel(obj1_lab)
    ax.set_ylabel(obj2_lab)
    ax.set_zlabel(obj3_lab)

    ax.scatter(objs_min[0], objs_min[1], objs_min[2], marker="*", c='orange', s=200)
    plt.show()
    return


def plot_runtime(nfe, metric, runtime_title, metric_label):
    """
    Generates a 2D plot showing showing how the metric value varies as NFE increases.

    Parameters
    ----------
    nfe : list
        List of NFE values according to step size.
    metric : list
        List of MOEA metric values associated with each NFE value.
    runtime_title : string
        Title of the plot.
    metric_label : string
        y-label of the plot.

    Returns
    -------
    None.

    """
    plt.plot(nfe, metric)
    plt.title(runtime_title)
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel(metric_label)
    plt.show()

    return

def runtime_hvol(algorithm, maxevals, frequency, file, hv):
    """
    Output runtime data for an algorithm run into a format readable by
    the MOEAFramework library

    Parameters
    ----------
    algorithm : MOEA type
        The MOEA to run and measure the hypervolume for.
    maxevals : int
        Maximum number of function evaluations (NFE).
    frequency : int
        NFE step size.
    file : arbitrary file
        Filename containing the output of the MOEA optimization.
    hv : function
        MOEAFramework function that takes two inputs:
            minimum (list): contains the minimum values of all objectives to be optimized.
            maximum (list): contains the maximum values of all objectives to be optimized.

    Returns
    -------
    nfe : list
        List of NFE values according to step size.
    hyp : list
        List of hypervolume values associated with each NFE value.

    """

    # open file and set up header
    f = open(file, "w+")
    f.write("# Variables = " + str(algorithm.problem.nvars))
    f.write("\n# Objectives = " + str(algorithm.problem.nobjs) + "\n")

    start_time = time.time()
    last_log = 0

    nvars = algorithm.problem.nvars
    nobjs = algorithm.problem.nobjs

    nfe = []
    hyp = []
    #front = []

    # run the algorithm/problem for specified number of function evaluations
    while (algorithm.nfe <= maxevals):
        # step the algorithm
        algorithm.step()
        #algorithm.run(algorithm.nfe + frequency)

        # print to file if necessary
        if (algorithm.nfe >= last_log + frequency):
            last_log = algorithm.nfe
            f.write("#\n//ElapsedTime=" +
                    str(datetime.timedelta(seconds=time.time()-start_time)))
            f.write("\n//NFE=" + str(algorithm.nfe) + "\n")

            arch = algorithm.archive[:]
            for i in range(len(arch)):
                sol = arch[i]
                for j in range(nvars):
                    f.write(str(sol.variables[j]) + " ")
                for j in range(nobjs):
                    f.write(str(sol.objectives[j]) + " ")
                f.write("\n")

            nfe.append(last_log)
            # use Platypus hypervolume indicator on the current archive
            result = hv.calculate(algorithm.archive[:])
            #result = hv.calc_internal(algorithm.archive[:], len(algorithm.archive[:]), 5)
            #print(f"currrent hyp = {result}")
            hyp.append(result)
    # close the runtime file
    end_time = time.time()
    total_time = end_time - start_time
    f.write("\nTime taken = " + str(total_time))
    f.close()
    return nfe, hyp

def select_objective(obj_name):
    """
    Selects the objective number for plotting on the 3D tradeoff surface given an objective name

    Parameters
    ----------
    obj_name : string
        Abbreviated name of the objective.

    Returns
    -------
    obj_num : int
        Index of the objective.
    obj_min : int
        Minimum value of the objective
    """
    obj_num = 0
    obj_min = 0
    if obj_name == 'Mean NPV':
        obj_num = 0
        obj_min = -6000
    elif obj_name == 'Mean prey deficit':
        obj_num = 1
        obj_min = 0
    elif obj_name == 'Mean WCLH':
        obj_num = 2
        obj_min = 0
    elif obj_name == 'Mean 1% harvest':
        obj_num = 3
        obj_min = -250
    elif obj_name == 'Mean harvest variance':
        obj_num = 4
        obj_min = 0

    return obj_num, obj_min



def plot_hvol(algorithm, maxevals, frequency, min_list, max_list, ax, pop_size=100):
    """
    Plots the hypervolume for up to NFE = maxevals

    Parameters
    ----------
    maxevals : int
        Maximum number of function evaluations.
    frequency : int
        NFE step size determined by the user .
    min_list : list
        Lower limit for all performance objectives.
    max_list : list
        Lower limit for all performance objectives.
    pop_size : int
        The population size for this randomly-generated seed

    Returns
    -------
    None.

    """
    # save the output for each population size as it's own unique output file
    output = "fishery_" + str(pop_size) + ".data"

    # set inputs for measuring hypervolume
    hv = Hypervolume(minimum=min_list, maximum=max_list)

    # Note: Cannot plot epsilon indicator and GD as those require reference sets, which
    # the fisheries problem does not have

    nfe, hyp = runtime_hvol(algorithm, maxevals, frequency, output, hv)

    # plot hypervolume
    #plot_runtime(nfe, hyp, 'PyBorg Runtime (Hypervolume)', 'Hypervolume')
    legend_label = "Pop. size=" + str(pop_size)
    r = random.uniform(0,1)
    g = random.uniform(0,1)
    b = random.uniform(0,1)
    rgb = (r,g,b)

    ax.plot(nfe, hyp, label=legend_label, color=rgb)

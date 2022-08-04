# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:07:57 2022

@author: lbl59

Code adapted from Antonia Hadjimichael's original code 
Author: Antonia Hadjimichael (hadjimichael@psu.edu)

Original code found in serial-borg-moea/Python/dtlz2_advanced.py 
Authors: Andrew Dircks & Dave Hadka

"""

import time
import datetime

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
#!/usr/bin/env python
#coding=utf-8

#======================================================================
# Program:     Neural Network Dose Distribution
# Author:      James Keal
# Date:        2016/09/12
#======================================================================

'''
comment
'''

from __future__ import print_function

import time
import numpy as np
import numpy.random as rnd

def all_coords_shuffled(dose, margin, n_samples=None):
    coord_list = []
    for d in range(len(dose)):
        for i in range(margin, dose[d].shape[0]-margin):
            for j in range(margin, dose[d].shape[1]-margin):
                for k in range(margin, dose[d].shape[2]-margin):
                    coord_list.append([d,i,j,k])

    assert n_samples <= len(coord_list)
    coord_list = np.array(coord_list)
    rnd.shuffle(coord_list)
    return coord_list[0:n_samples]

def monte_carlo(dose, margin, n_samples):
    coord_list = []

    for n in range(n_samples):
        d = rnd.choice(len(dose))
        r = rnd.rand(3)
        i = int(r[0]*(dose[d].shape[0]-2*margin)) + margin
        j = int(r[1]*(dose[d].shape[1]-2*margin)) + margin
        k = int(r[2]*(dose[d].shape[2]-2*margin)) + margin
        coord_list.append([d,i,j,k])

    coord_list = np.array(coord_list)
    return coord_list

def latin_hypercube(dose, margin, n_samples):
    start_time = time.time()
    coord_list = []
    print(time.time() - start_time)
    return coord_list

def guassian_latin_hypercube(dose, margin, n_samples):
    start_time = time.time()
    coord_list = []
    print(time.time() - start_time)
    return coord_list

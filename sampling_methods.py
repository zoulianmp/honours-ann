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
import random as rnd

def all_coords_shuffled(dose, margin, n_samples=None):
    coord_list = []
    for d in range(len(dose)):
        for i in range(margin, dose[d].shape[0] - margin):
            for j in range(margin, dose[d].shape[1] - margin):
                for k in range(margin, dose[d].shape[2] - margin):
                    coord_list.append([d,i,j,k])

    assert n_samples <= len(coord_list)
    rnd.shuffle(coord_list)
    return coord_list[0:n_samples]

def monte_carlo(dose, margin, n_samples):
    coord_list = []

    for n in range(n_samples):
        d = rnd.choice(range(len(dose)))
        i = int(rnd.random()*(dose[d].shape[0] - 2*margin)) + margin
        j = int(rnd.random()*(dose[d].shape[1] - 2*margin)) + margin
        k = int(rnd.random()*(dose[d].shape[2] - 2*margin)) + margin
        coord_list.append([d,i,j,k])

    return coord_list

def latin_hypercube(dose, margin, n_samples, shuffle=True):
    coord_list = []
    n_per_dose = int(n_samples/len(dose))

    for d in range(len(dose)):
        n_min_dim = min(dose[d].shape) - 2*margin
        n_batch = int(n_per_dose/n_min_dim)
        n_rem = n_per_dose % n_min_dim

        for b in range(n_batch + 1):
            x = range(dose[d].shape[0] - 2*margin)
            y = range(dose[d].shape[1] - 2*margin)
            z = range(dose[d].shape[2] - 2*margin)
            if b == n_batch:
                x = rnd.sample(x, n_rem)
                y = rnd.sample(y, n_rem)
                z = rnd.sample(z, n_rem)
            else:
                x = rnd.sample(x, n_min_dim)
                y = rnd.sample(y, n_min_dim)
                z = rnd.sample(z, n_min_dim)
            for i,j,k in zip(x,y,z):
                coord_list.append([d,i,j,k])

    if shuffle:
        rnd.shuffle(coord_list)
    return coord_list

def weighted_latin_hypercube(dose, margin, n_samples, shuffle=True):
    coord_list = []
    n_per_dose = int(n_samples/len(dose))

    for d in range(len(dose)):
        n_min_dim = min(dose[d].shape) - 2*margin
        n_min_dim = int(n_min_dim/2)
        n_batch = int(n_per_dose/n_min_dim)
        n_rem = n_per_dose % n_min_dim

        for b in range(n_batch + 1):
            x = range(dose[d].shape[0] - 2*margin)
            y = range(dose[d].shape[1] - 2*margin)
            lx, ly = int(len(x)/3), int(len(y)/3)

            # 3 times more likely to choose central third
            x = x[:lx] + x[lx:-lx] + x[lx:-lx] + x[lx:-lx] + x[-lx:]
            y = y[:ly] + y[ly:-ly] + y[ly:-ly] + y[ly:-ly] + y[-ly:]
            z = range(dose[d].shape[2] - 2*margin)
            if b == n_batch:
                x = rnd.sample(x, n_rem)
                y = rnd.sample(y, n_rem)
                z = rnd.sample(z, n_rem)
            else:
                x = rnd.sample(x, n_min_dim)
                y = rnd.sample(y, n_min_dim)
                z = rnd.sample(z, n_min_dim)
            for i,j,k in zip(x,y,z):
                coord_list.append([d,i,j,k])

    if shuffle:
        rnd.shuffle(coord_list)
    return coord_list

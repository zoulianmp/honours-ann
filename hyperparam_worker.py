#!/usr/bin/env python
#coding=utf-8

#======================================================================
# Program:     Neural Network Dose Distribution
# Author:      James Keal
# Date:        2016/08/04
#======================================================================

'''
comment
'''

from __future__ import print_function

import os
import sys
import time
import datetime
import imp
import mhd_utils_3d as mhd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
import train_dose_net as train
import random as rnd

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]

def main(ann3d, density_dir, intgr_dir, fluence_dir, dose_dir, output_file=None):
    for i in range(1000):
        num_epochs = 100
        batchsize = int(512 * rnd.random() + 64)
        #momentum = 0.3 * rnd.random() + 0.7
        momentum = 'N/A'
        learning_rate = rnd.random() * 10**(-4-2*rnd.random())
        learning_rate = learning_rate * batchsize/256

        train.main(ann3d, density_dir, intgr_dir, fluence_dir, dose_dir,
                        output_file,
                            [learning_rate, batchsize, num_epochs, momentum])


if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 6):
        print("Trains a neural network to predict radiation dose.")
        print("Usage: %s <PARAMS> <DENSITY> <INTEGRALS> <FLUENCE> <DOSE> [OUTPUT_FILE]"
                % sys.argv[0])
        print()
        print("PARAMS: A python module containing the parameters of learning,")
        print("    network architecture, and data sampling methods employed.")
        print("DENSITY: The path to a folder containing 'n' MHD files, each")
        print("    containing the voxel densities of a phantom.")
        print("INTEGRALS: The path to a folder containing 'n' MHD files, each")
        print("    containing the integral densities of a phantom.")
        print("FLUENCE: The path to a folder containing 'n' MHD files, each")
        print("    containing the voxel fluences of a phantom.")
        print("DOSE: The path to a folder containing 'n' MHD files, each")
        print("    containing the voxel doses (targets) of a phantom.")
        print("OUTPUT_FILE: The name of the MHD file that will contain the")
        print("    generated data.")
    else:
        kwargs = {}
        kwargs['ann3d'] = sys.argv[1]
        kwargs['density_dir'] = sys.argv[2]
        kwargs['intgr_dir'] = sys.argv[3]
        kwargs['fluence_dir'] = sys.argv[4]
        kwargs['dose_dir'] = sys.argv[5]
        if len(sys.argv) > 6:
            kwargs['output_file'] = sys.argv[6]
        main(**kwargs)

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

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]

def main(model):
    # load the network weights from file
    print('\n')
    print('Importing network from file...')
    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    print(len(param_values))
    print(param_values[0].shape)

    # Make plots
    now = datetime.datetime.now()
    now = str(now.month) + '-' + str(now.day) + '_' + \
          str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)

    #DOSE = 1

    #DOSE = DOSE - param_values[-1]
    #DOSE = np.dot(param_values[-2],DOSE)
    #DOSE = DOSE - param_values[-3]
    #DOSE = np.dot(param_values[-4],DOSE)
    #DOSE = DOSE - param_values[-5]
    #DOSE = np.dot(param_values[-6],DOSE)

    #print(DOSE.shape)

    im = plt.figure()
    array = param_values[0]
    node = np.sum(array[:-3,:], axis=1)
    print(node.shape)
    node = np.reshape(node, (3,7,7,7))
    slic = np.sum(node[0,:,:,:], axis=0)
    plt.plot(slic)
    #plt.colorbar()
    im.savefig('image_' + now + '(' + str(i) + ').png')

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 2):
        print("Foo.")
    else:
        kwargs = {}
        kwargs['model'] = sys.argv[1]
        main(**kwargs)

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

def main(ann3d, model, density_dir, intgr_dir, fluence_dir, output_file=None):
    if output_file is None:
        now = datetime.datetime.now()
        now = str(now.month) + '-' + str(now.day) + '_' + \
              str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        output_file = 'prediction_' + now + '.mhd'

    print('\n')
    print('Loading network parameters...')
    ann3d = imp.load_source('networks.ann3d', ann3d)

    # Load a set of volumes from multiple files
    print('\n')
    print('Loading density data...')
    density = []
    for dirpath, subdirs, files in os.walk(density_dir):
        for f in sorted(files):
            if f.endswith(".mhd"):
                print(f)
                density += [mhd.load_mhd(os.path.join(dirpath, f))[0]]

    # Load a set of volumes from multiple files
    print('\n')
    print('Loading integral data...')
    integral = []
    for dirpath, subdirs, files in os.walk(intgr_dir):
        for f in sorted(files):
            if f.endswith(".mhd"):
                print(f)
                integral += [mhd.load_mhd(os.path.join(dirpath, f))[0]]

    # Load a set of volumes from multiple files
    print('\n')
    print('Loading fluence data...')
    fluence = []
    for dirpath, subdirs, files in os.walk(fluence_dir):
        for f in sorted(files):
            if f.endswith(".mhd"):
                print(f)
                fluence += [mhd.load_mhd(os.path.join(dirpath, f))[0]]

    # load the network weights from file
    print('\n')
    print('Importing network from file...')
    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    print('\n')
    print('Building model and compiling functions...')
    network = ann3d.ann3d_model(input_var)
    lasagne.layers.set_all_param_values(network, param_values)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    feed_forward = theano.function([input_var], test_prediction)

    # Deploy!
    prediction = []
    for i in range(len(fluence)):
        prediction += [ann3d.ann3d_deploy(density[i], integral[i], fluence[i], feed_forward)]
        mhd.write_mhd('(' + str(i) + ')' + output_file,
                prediction[i], prediction[i].shape, VOXEL_SIZE)

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 6):
        print("Stores the predicted dose for an entire volume.")
        print("Usage: %s <PARAMS> <MODEL> <DENSITY> <FLUENCE> [OUTPUT_FILE]"
                % sys.argv[0])
        print()
        print("PARAMS: A python module containing the parameters of learning,")
        print("    network architecture, and data sampling methods employed.")
        print("MODEL: The path to an NPZ file containing the network weights.")
        print("DENSITY: The path to a folder containing 'n' MHD files, each")
        print("    containing the voxel densities of a phantom.")
        print("INTEGRALS: The path to a folder containing 'n' MHD files, each")
        print("    containing the integral densities of a phantom.")
        print("FLUENCE: The path to a folder containing 'n' MHD files, each")
        print("    containing the voxel fluences of a phantom.")
        print("OUTPUT_FILE: The name of the MHD file that will contain the")
        print("    integral data.")
    else:
        kwargs = {}
        kwargs['ann3d'] = sys.argv[1]
        kwargs['model'] = sys.argv[2]
        kwargs['density_dir'] = sys.argv[3]
        kwargs['intgr_dir'] = sys.argv[4]
        kwargs['fluence_dir'] = sys.argv[5]
        if len(sys.argv) > 6:
            kwargs['output_file'] = sys.argv[6]
        main(**kwargs)

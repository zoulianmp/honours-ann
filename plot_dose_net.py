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

import sys
import time
import mhd_utils_3d as mhd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne

from ann3d_kalantzis import *
from ann3d_keal import *

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]
FIELD_SIZE = (( 3.0, 3.0),
              ( 5.0, 5.0),
              (10.0,10.0))          # field size (x,y) [cm]
SS_DIST = 100.0                     # source-surface distance [cm]

MASS_ATT_AIR = 2.522e-2         # mass attenuation of air [cm^2/g]
MASS_ATT_WAT = 2.770e-2         # mass attenuation of water [cm^2/g]
DENSITY_AIR = 1.225e-3          # density of air [g/cm^3]
DENSITY_WAT = 1.000             # density of water [g/cm^3]

def generate_fluence(vol_shape, vox_size, field_size):
    fluence = np.zeros(vol_shape)

    z = vox_size[0] * np.arange(vol_shape[0]-1,-1,-1) + SS_DIST
    y = vox_size[1] * np.concatenate(
            (np.arange(vol_shape[1]/2-1,-1,-1), np.arange(0, vol_shape[1]/2)))
    x = vox_size[2] * np.concatenate(
            (np.arange(vol_shape[2]/2-1,-1,-1), np.arange(0, vol_shape[2]/2)))

    Z,Y,X = np.meshgrid(z,y,x, indexing='ij')
    R_sq = np.square(X) + np.square(Y) + np.square(Z)
    R = np.sqrt(R_sq)

    R_air = R*(SS_DIST/Z)
    R_wat = R - R_air

    tau_air = R_air*MASS_ATT_AIR*DENSITY_AIR
    tau_wat = R_wat*MASS_ATT_WAT*DENSITY_WAT

    fluence = 1.0*np.exp(-tau_air-tau_wat)/R_sq
    fluence = fluence*np.less_equal(X,field_size[0]*Z/(2.0*SS_DIST))
    fluence = fluence*np.less_equal(Y,field_size[1]*Z/(2.0*SS_DIST))

    return fluence/np.max(fluence)

def main(filename, model='keal'):
    print('\n')
    print('Loading dose data...')

    # Load a dose volume from a single file
    #dose = [mhd.load_mhd('data/combined_5cm_water_energy.mhd')[0]]

    # Or a set of volumes from multiple files
    dose = []
    dose += [mhd.load_mhd('energy/combined_3cm_water_energy.mhd')[0]]
    dose += [mhd.load_mhd('energy/combined_5cm_water_energy.mhd')[0]]
    dose += [mhd.load_mhd('energy/combined_10cm_water_energy.mhd')[0]]

    # Calculate the fluence for each field size
    print('\n')
    print('Calculating fluence...')
    fluence = []
    for i in range(len(FIELD_SIZE)):
        fluence += [generate_fluence(dose[i].shape, VOXEL_SIZE, FIELD_SIZE[i])]

    # load the network weights from file
    print('\n')
    print('Importing network from file...')
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    if model == 'k3d':
        print('\n')
        print('Building "kalantzis" model and compiling functions...')
        network = ann3d_kalantzis_model(input_var)
        lasagne.layers.set_all_param_values(network, param_values)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        feed_forward = theano.function([input_var], test_prediction)
        ann3d_kalantzis_plot_pdd(dose[0], fluence[0], feed_forward)
        ann3d_kalantzis_plot_profile(dose[0], fluence[0], feed_forward)
    elif model == 'keal':
        print('\n')
        print('Building "keal" model and compiling functions...')
        network = ann3d_keal_model(input_var)
        lasagne.layers.set_all_param_values(network, param_values)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        feed_forward = theano.function([input_var], test_prediction)
        ann3d_keal_plot_pdd(dose[0], fluence[0], feed_forward)
        ann3d_keal_plot_profile(dose[0], fluence[0], feed_forward)
    else:
        print("Unrecognized model type %r." % model)
        return

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Does something...")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['filename'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['model'] = sys.argv[2]
        main(**kwargs)

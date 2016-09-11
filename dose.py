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
import os
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
FIELD_SIZE = (5.0,5.0)              # field size (x,y) [cm]
SS_DIST = 100.0                     # source-surface distance [cm]

MASS_ATT_AIR = 2.522e-2         # mass attenuation of air [cm^2/g]
MASS_ATT_WAT = 2.770e-2         # mass attenuation of water [cm^2/g]
DENSITY_AIR = 1.225e-3          # density of air [g/cm^3]
DENSITY_WAT = 1.000             # density of water [g/cm^3]

def import_dose(root_dir, unit='dose'):
    if unit == 'dose':
        unit = '/output-Dose.mhd'
    elif unit == 'energy':
        unit = '/output-Edep.mhd'
    else:
        print("Unrecognized measurement type %r." % unit)
        return

    dirs = os.walk(root_dir).next()[1]
    n_dir = 0
    for path in dirs:
        n_dir += 1
        ddata = mhd.load_mhd(root_dir + '/' + path + unit)[0]
        if n_dir == 1:
            data = ddata
        else:
            data += ddata

    return data

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

    return fluence

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(model='keal', num_epochs=100, batchsize=500):
    # Load the dose data and calculate the fluence
    print('\n')
    print('Loading dose data...')
    dose = import_dose('5cm', 'energy')
    dose = dose/np.max(dose)

    print('\n')
    print('Calculating fluence...')
    fluence = generate_fluence(dose.shape, VOXEL_SIZE, FIELD_SIZE)
    fluence = fluence/np.max(fluence)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    if model == 'k3d':
        print('\n')
        print('Preparing "kalantzis" data set...')
        x_train, y_train, x_val, y_val, x_test, y_test = ann3d_kalantzis_dataset(dose, fluence)
        print('\n')
        print('Building "kalantzis" model and compiling functions...')
        network = ann3d_kalantzis_model(input_var)
    if model == 'keal':
        print('\n')
        print('Preparing "keal" data set...')
        x_train, y_train, x_val, y_val, x_test, y_test = ann3d_keal_dataset(dose, fluence)
        print('\n')
        print('Building "keal" model and compiling functions...')
        network = ann3d_keal_model(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.07)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = target_var

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    feed_forward = theano.function([input_var], test_prediction)

    # Finally, launch the training loop.
    print('\n')
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, batchsize, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc = acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    predicted_dose = []
    for i in range(1,dose.shape[0]-1):
        x_in = [[[[
            fluence[i,fluence.shape[1]/2,fluence.shape[2]/2],
            fluence[i-1,fluence.shape[1]/2,fluence.shape[2]/2],
            fluence[i+1,fluence.shape[1]/2,fluence.shape[2]/2],
            fluence[i,fluence.shape[1]/2-1,fluence.shape[2]/2],
            fluence[i,fluence.shape[1]/2+1,fluence.shape[2]/2],
            fluence[i,fluence.shape[1]/2,fluence.shape[2]/2-1],
            fluence[i,fluence.shape[1]/2,fluence.shape[2]/2+1],
            float(i)/fluence.shape[0],
            0,
            0
            ]]]]
        x_in = np.array(x_in, dtype=np.float32)
        ff = (feed_forward(x_in))
        predicted_dose.append(ff[0])
    plt.plot(range(0,dose.shape[0]), dose[:,dose.shape[1]/2,dose.shape[2]/2],
             range(1,dose.shape[0]-1), predicted_dose)
    plt.show()

    predicted_dose = []
    for i in range(1,dose.shape[1]-1):
        x_in = [[[[
            fluence[fluence.shape[0]/2,i,fluence.shape[2]/2],
            fluence[fluence.shape[0]/2-1,i,fluence.shape[2]/2],
            fluence[fluence.shape[0]/2+1,i,fluence.shape[2]/2],
            fluence[fluence.shape[0]/2,i-1,fluence.shape[2]/2],
            fluence[fluence.shape[0]/2,i+1,fluence.shape[2]/2],
            fluence[fluence.shape[0]/2,i,fluence.shape[2]/2-1],
            fluence[fluence.shape[0]/2,i,fluence.shape[2]/2+1],
            0.5,
            float(i)/fluence.shape[1] - 0.5,
            0
            ]]]]
        x_in = np.array(x_in, dtype=np.float32)
        ff = (feed_forward(x_in))
        predicted_dose.append(ff[0])
    plt.plot(range(0,dose.shape[1]), dose[dose.shape[0]/2,:,dose.shape[2]/2],
             range(1,dose.shape[1]-1), predicted_dose)
    plt.show()

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


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
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)

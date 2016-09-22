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
import sampling_methods as sm
import theano
import theano.tensor as T
import lasagne

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]

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

def main(ann3d, density_dir, fluence_dir, dose_dir, output_file=None):
    print('\n')
    print('Loading network parameters...')
    ann3d = imp.load_source('networks.ann3d', ann3d)
    learning_rate = ann3d.LEARNING_RATE
    batchsize = ann3d.BATCHSIZE
    num_epochs = ann3d.NUM_EPOCHS
    momentum = ann3d.MOMENTUM

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
    print('Loading fluence data...')
    fluence = []
    for dirpath, subdirs, files in os.walk(fluence_dir):
        for f in sorted(files):
            if f.endswith(".mhd"):
                print(f)
                fluence += [mhd.load_mhd(os.path.join(dirpath, f))[0]]

    # Load a set of volumes from multiple files
    print('\n')
    print('Loading dose data...')
    dose = []
    for dirpath, subdirs, files in os.walk(dose_dir):
        for f in sorted(files):
            if f.endswith(".mhd"):
                print(f)
                dose += [mhd.load_mhd(os.path.join(dirpath, f))[0]]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    print('\n')
    print('Sampling data set...')
    x_train, y_train, x_val, y_val, x_test, y_test = ann3d.ann3d_dataset(dose, fluence)

    print('\n')
    print('Building model and compiling functions...')
    network = ann3d.ann3d_model(input_var)

    # Create a loss expression for training, i.e., a scalar objective:
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean().sqrt()
    # We could add some weight decay as well here

    # Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean().sqrt()
    feed_forward = theano.function([input_var], test_prediction)

    # Compile a function performing a training step on a mini-batch:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a function to compute the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)

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
        val_batches = 0
        for batch in iterate_minibatches(x_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6e}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6e}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6e}".format(test_err / test_batches))

    # Now dump the network weights to a file:
    now = datetime.datetime.now()
    now = str(now.month) + '-' + str(now.day) + '_' + \
          str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
    np.savez('model_' + now, *lasagne.layers.get_all_param_values(network))

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 5):
        print("Trains a neural network to predict radiation dose.")
        print("Usage: %s <PARAMS> <DENSITY> <FLUENCE> <DOSE> [OUTPUT_FILE]"
                % sys.argv[0])
        print()
        print("PARAMS: A python module containing the parameters of learning,")
        print("    network architecture, and data sampling methods employed.")
        print("DENSITY: The path to a folder containing 'n' MHD files, each")
        print("    containing the voxel densities of a phantom.")
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
        kwargs['fluence_dir'] = sys.argv[3]
        kwargs['dose_dir'] = sys.argv[4]
        if len(sys.argv) > 5:
            kwargs['output_file'] = sys.argv[5]
        main(**kwargs)

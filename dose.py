#!/usr/bin/env python
#coding=utf-8

#======================================================================
# Program:     Neural Network Dose Distribution
# Language:    Python
# Author:      James Keal
# Date:        2016/08/04
# Last Edited: Keal 2016/08/12
#======================================================================

'''
comment
'''

from __future__ import print_function

import sys
import os
import time
#import math

import mhd_utils_3d as mhd

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]
FIELD_SIZE = (5.0,5.0)              # field size (x,y) [cm]
SS_DIST = 100.0                     # source-surface distance [cm]

MASS_ATT_AIR = 2.522e-2         # mass attenuation of air [cm^2/g]
MASS_ATT_WAT = 2.770e-2         # mass attenuation of water [cm^2/g]
DENSITY_AIR = 1.225e-3          # density of air [g/cm^3]
DENSITY_WAT = 1.000             # density of water [g/cm^3]

# Print iterations progress
def pp(iteration, total):
    prefix = 'Progress:'
    suffix = 'Complete'
    barLength = 32
    decimals = 2
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.0 * (iteration / float(total)), decimals)
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %(prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def import_dose(root_dir):
    dirs = os.walk(root_dir).next()[1]
    n_dir = 0
    print('\n')
    print('Loading dose data...')
    pp(n_dir, len(dirs))
    for path in dirs:
        n_dir += 1
        pp(n_dir, len(dirs))
        dd = mhd.load_mhd(root_dir + '/' + path + '/output-Dose.mhd')[0]
        if n_dir == 1:
            data = dd
        else:
            data += dd
    return data

def generate_fluence(vol_shape, vox_size, field_size):
    fluence = np.zeros(vol_shape)
    print('\n')
    print('Calculating fluence...')
    pp(0,11)

    z = vox_size[0] * np.arange(vol_shape[0]-1,-1,-1) + SS_DIST
    y = vox_size[1] * np.concatenate(
            (np.arange(vol_shape[1]/2-1,-1,-1), np.arange(0, vol_shape[1]/2)))
    x = vox_size[2] * np.concatenate(
            (np.arange(vol_shape[2]/2-1,-1,-1), np.arange(0, vol_shape[2]/2)))
    pp(1,11)

    Z,Y,X = np.meshgrid(z,y,x, indexing='ij')
    pp(2,11)
    R_sq,__ = np.square(X) + np.square(Y) + np.square(Z)                    ,pp(3,11)
    R,__ = np.sqrt(R_sq)                                                    ,pp(4,11)

    R_air,__ = R*(SS_DIST/Z)                                                ,pp(5,11)
    R_wat,__ = R - R_air                                                    ,pp(6,11)

    tau_air,__ = R_air*MASS_ATT_AIR*DENSITY_AIR                             ,pp(7,11)
    tau_wat,__ = R_wat*MASS_ATT_WAT*DENSITY_WAT                             ,pp(8,11)

    fluence,__ = 1.0*np.exp(-tau_air-tau_wat)/R_sq                          ,pp(9,11)
    fluence,__ = fluence*np.less_equal(X,field_size[0]*Z/(2.0*SS_DIST))     ,pp(10,11)
    fluence,__ = fluence*np.less_equal(Y,field_size[1]*Z/(2.0*SS_DIST))     ,pp(11,11)

    return fluence

def load_kalantzis_3d_dataset(dose, fluence):
    assert dose.shape == fluence.shape
    print('\n')
    print('Preparing data set...')
    pp(0,6)

    # The outer voxels of the dose tensors are unusable since the neighbouring
    # voxels are included as input to the neural network. We create a set of
    # all usable coordinates:
    coord_list = []
    for i in range(1, dose.shape[0]-1):
        for j in range(1, dose.shape[1]-1):
            for k in range(1, dose.shape[2]-1):
                coord_list.append([i,j,k])

    pp(1,6)
    coord_list = np.array(coord_list)
    np.random.shuffle(coord_list)
    pp(2,6)


    x_tmp = np.zeros((len(coord_list),1,1,7))
    y_tmp = np.zeros((len(coord_list),1))
    for i in range(len(coord_list)):
        c = coord_list[i]
        x_tmp[i][0][0] = [
            fluence[c[0],c[1],c[2]],
            fluence[c[0]-1,c[1],c[2]],
            fluence[c[0]+1,c[1],c[2]],
            fluence[c[0],c[1]-1,c[2]],
            fluence[c[0],c[1]+1,c[2]],
            fluence[c[0],c[1],c[2]-1],
            fluence[c[0],c[1],c[2]+1]
            ]
        y_tmp[i] = [dose[c[0],c[1],c[2]]]
    pp(5,6)

    # Use 70% of usable coordinates as training data, 15% as valaidation
    # data and 15% as testing data.
    n_train = int(0.70 * len(coord_list))
    n_test =  int(0.15 * len(coord_list))

    x_train = np.array(x_tmp[:n_train], dtype=np.float32)
    y_train = np.array(y_tmp[:n_train], dtype=np.float32)
    x_test = np.array(x_tmp[-n_test:], dtype=np.float32)
    y_test = np.array(y_tmp[-n_test:], dtype=np.float32)
    x_val = np.array(x_tmp[n_train:-n_test], dtype=np.float32)
    y_val = np.array(y_tmp[n_train:-n_test], dtype=np.float32)
    pp(6,6)

    # (It doesn't matter how we do this as long as we can read them again.)
    return x_train, y_train, x_val, y_val, x_test, y_test

def kalantzis_3d(input_var=None):
    print('\n')
    print("Building model and compiling functions...")
    pp(0,7)
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 7 rows and 1 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 7, 1), input_var=input_var)
    pp(1,7)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    pp(2,7)

    # Add a fully-connected layer of 30 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=30,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    pp(3,7)

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    pp(4,7)

    # A 5-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=5,
            nonlinearity=lasagne.nonlinearities.rectify)
    pp(5,7)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    pp(6,7)

    # Finally, we'll add the fully-connected output layer, of 1 rectifier unit:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)
    pp(7,7)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

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

def main(model='k3d', num_epochs=500):
    # Load the dose data and calculate the fluence
    dose = import_dose('5cm')
    fluence = generate_fluence(dose.shape, VOXEL_SIZE, FIELD_SIZE)

    x_train, y_train, x_val, y_val, x_test, y_test = load_kalantzis_3d_dataset(dose, fluence)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    if model == 'k3d':
        network = kalantzis_3d(input_var)
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
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        print('TRAIN')
        for batch in iterate_minibatches(x_train, y_train, 600, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        print('VALIDATE')
        for batch in iterate_minibatches(x_val, y_val, 600, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

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

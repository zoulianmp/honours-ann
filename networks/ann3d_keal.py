#!/usr/bin/env python
#coding=utf-8

#======================================================================
# Program:     Neural Network Dose Distribution
# Author:      James Keal
# Date:        2016/09/11
#======================================================================

'''
comment
'''

import numpy as np
import sampling_methods as sm
import matplotlib.pyplot as plt
import lasagne

LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCHSIZE = 500
MOMENTUM = 0.07

INPUT_MARGIN = 2
OUTPUT_MARGIN = 0

def ann3d_dataset(dose, fluence):
    # The outer voxels of the dose tensors are unusable since the neighbouring
    # voxels are included as input to the neural network. We create a set of
    # all usable coordinates:
    coord_list = sm.monte_carlo(dose, INPUT_MARGIN, 4194304)

    x_list = np.zeros((len(coord_list), 1, 1, (2*INPUT_MARGIN+1)**3))
    y_list = np.zeros((len(coord_list),(2*OUTPUT_MARGIN+1)**3))
    for i in range(len(coord_list)):
        c = coord_list[i]
        x_tmp = []
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_tmp += [fluence[c[0]][c[1]+ii,c[2]+jj,c[3]+kk]]

        y_tmp = []
        for ii in range(-OUTPUT_MARGIN, OUTPUT_MARGIN+1):
            for jj in range(-OUTPUT_MARGIN, OUTPUT_MARGIN+1):
                for kk in range(-OUTPUT_MARGIN, OUTPUT_MARGIN+1):
                    y_tmp += [dose[c[0]][c[1]+ii,c[2]+jj,c[3]+kk]]

        x_list[i][0][0] = x_tmp
        y_list[i] = y_tmp

    # Use 70% of usable coordinates as training data, 15% as valaidation
    # data and 15% as testing data.
    n_train = int(0.70 * len(coord_list))
    n_test =  int(0.15 * len(coord_list))

    x_train = np.array(x_list[:n_train], dtype=np.float32)
    y_train = np.array(y_list[:n_train], dtype=np.float32)
    x_test = np.array(x_list[-n_test:], dtype=np.float32)
    y_test = np.array(y_list[-n_test:], dtype=np.float32)
    x_val = np.array(x_list[n_train:-n_test], dtype=np.float32)
    y_val = np.array(y_list[n_train:-n_test], dtype=np.float32)

    # (It doesn't matter how we do this as long as we can read them again.)
    return x_train, y_train, x_val, y_val, x_test, y_test

def ann3d_model(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 1 rows and many columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, (2*INPUT_MARGIN+1)**3),
                                        input_var=input_var)

    # Apply 0% dropout to the input data:
    #l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.0)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 20%:
    #l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 1 rectifier unit:
    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=(2*OUTPUT_MARGIN+1)**3,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def ann3d_plot_pdd(dose, fluence, feed_forward):
    predicted_dose = []
    for i in range(INPUT_MARGIN, dose.shape[0]-INPUT_MARGIN):
        x_in = [[[[]]]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_in[0][0][0] += [fluence[
                            i + ii,
                            fluence.shape[1]/2 + jj,
                            fluence.shape[2]/2 + kk
                            ]]

        x_in = np.array(x_in, dtype=np.float32)
        ff = feed_forward(x_in)
        predicted_dose.append(ff[0])
    fig = plt.figure()
    plt.plot(range(0,dose.shape[0]), dose[:,dose.shape[1]/2,dose.shape[2]/2],
             range(INPUT_MARGIN,dose.shape[0]-INPUT_MARGIN), predicted_dose)
    return fig

def ann3d_plot_profile(dose, fluence, feed_forward):
    predicted_dose = []
    for i in range(INPUT_MARGIN, dose.shape[1]-INPUT_MARGIN):
        x_in = [[[[]]]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_in[0][0][0] += [fluence[
                            fluence.shape[0]/2 + ii,
                            i + jj,
                            fluence.shape[2]/2 + kk
                            ]]

        x_in = np.array(x_in, dtype=np.float32)
        ff = feed_forward(x_in)
        predicted_dose.append(ff[0])
    fig = plt.figure()
    plt.plot(range(0,dose.shape[1]), dose[dose.shape[0]/2,:,dose.shape[2]/2],
             range(INPUT_MARGIN,dose.shape[1]-INPUT_MARGIN), predicted_dose)
    return fig

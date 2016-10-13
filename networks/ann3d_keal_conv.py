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
import lasagne.layers.dnn

# hyperparameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCHSIZE = 512
MOMENTUM = 0.9
N_SAMPLES = 262144

INPUT_MARGIN = 8
OUTPUT_MARGIN = 0
PRECISION = np.float16

def ann3d_dataset(density, integral, fluence, dose):
    # The outer voxels of the dose tensors are unusable since the neighbouring
    # voxels are included as input to the neural network. We create a set of
    # susable coordinates:
    coord_list = sm.monte_carlo(dose, INPUT_MARGIN, N_SAMPLES)

    n_side = 2*INPUT_MARGIN + 1
    x_list = np.empty((len(coord_list), 3, n_side, n_side, n_side), dtype=PRECISION)
    y_list = np.empty((len(coord_list), 1), dtype=PRECISION)
    for i in range(len(coord_list)):
        c = coord_list[i]

        x_list[i][0] = density[c[0]][c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                                     c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1,
                                     c[3]-INPUT_MARGIN:c[3]+INPUT_MARGIN+1]
        x_list[i][1] = integral[c[0]][c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                                      c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1,
                                      c[3]-INPUT_MARGIN:c[3]+INPUT_MARGIN+1]
        x_list[i][2] = fluence[c[0]][c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                                     c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1,
                                     c[3]-INPUT_MARGIN:c[3]+INPUT_MARGIN+1]

        y_list[i] = dose[c[0]][c[1], c[2], c[3]]

    # Use 70% of usable coordinates as training data, 15% as valaidation
    # data and 15% as testing data.
    n_train = int(0.70 * len(coord_list))
    n_test =  int(0.15 * len(coord_list))

    x_train = x_list[:n_train]
    y_train = y_list[:n_train]
    x_test = x_list[-n_test:]
    y_test = y_list[-n_test:]
    x_val = x_list[n_train:-n_test]
    y_val = y_list[n_train:-n_test]

    # (It doesn't matter how we do this as long as we can read them again.)
    return x_train, y_train, x_val, y_val, x_test, y_test

def ann3d_model(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 1 rows and many columns) and
    # linking it to the given Theano variable `input_var`, if any:
    n_side = 2*INPUT_MARGIN + 1
    network = lasagne.layers.InputLayer(shape=(None, 3, n_side, n_side, n_side),
                                        input_var=input_var)

    network = lasagne.layers.GaussianNoiseLayer(network, sigma=0.1)

    # Apply 0% dropout to the input data:
    #network = lasagne.layers.DropoutLayer(network, p=0.0)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    network = lasagne.layers.dnn.Conv3DDNNLayer(
            network, num_filters=5, filter_size=(5,5,5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2,2,2))

    # We'll now add dropout of 20%:
    #network = lasagne.layers.DropoutLayer(network, p=0.2)

    network = lasagne.layers.DenseLayer(
            network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 1 rectifier unit:
    network = lasagne.layers.DenseLayer(
            network, num_units=(2*OUTPUT_MARGIN+1)**3,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return network

def ann3d_plot_pdd(density, integral, fluence, dose, feed_forward):
    predicted_dose = []
    for i in range(INPUT_MARGIN, dose.shape[0]-INPUT_MARGIN):
        c = [i, dose.shape[1]/2, dose.shape[2]/2]

        x_list = [[0,0,0]]
        x_list[0][0] = density[c[0]-INPUT_MARGIN:c[0]+INPUT_MARGIN+1,
                               c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                               c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1]
        x_list[0][1] = integral[c[0]-INPUT_MARGIN:c[0]+INPUT_MARGIN+1,
                                c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                                c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1]
        x_list[0][2] = fluence[c[0]-INPUT_MARGIN:c[0]+INPUT_MARGIN+1,
                               c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                               c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1]

        x_list = np.array(x_list, dtype=PRECISION)
        ff = feed_forward(x_list)
        predicted_dose.append(ff[0])
    fig = plt.figure()
    plt.plot(range(0,dose.shape[0]), dose[:,dose.shape[1]/2,dose.shape[2]/2], label="real")
    plt.plot(range(INPUT_MARGIN,dose.shape[0]-INPUT_MARGIN), predicted_dose, label="ann")
    plt.legend()
    return fig

def ann3d_plot_profile(density, integral, fluence, dose, feed_forward):
    predicted_dose = []
    for i in range(INPUT_MARGIN, dose.shape[2]-INPUT_MARGIN):
        c = [density.shape[0]/2, density.shape[1]/2, i]

        x_list = [[0,0,0]]
        x_list[0][0] = density[c[0]-INPUT_MARGIN:c[0]+INPUT_MARGIN+1,
                               c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                               c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1]
        x_list[0][1] = integral[c[0]-INPUT_MARGIN:c[0]+INPUT_MARGIN+1,
                                c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                                c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1]
        x_list[0][2] = fluence[c[0]-INPUT_MARGIN:c[0]+INPUT_MARGIN+1,
                               c[1]-INPUT_MARGIN:c[1]+INPUT_MARGIN+1,
                               c[2]-INPUT_MARGIN:c[2]+INPUT_MARGIN+1]

        x_list = np.array(x_list, dtype=PRECISION)
        ff = feed_forward(x_list)
        predicted_dose.append(ff[0])
    fig = plt.figure()
    plt.plot(range(0,dose.shape[2]), dose[dose.shape[0]/2,dose.shape[1]/2,:], label="real")
    plt.plot(range(INPUT_MARGIN,dose.shape[2]-INPUT_MARGIN), predicted_dose, label="ann")
    plt.legend()
    return fig

def ann3d_deploy(density, integral, fluence, feed_forward):
    predicted_dose = np.empty(fluence.shape)
    it = np.nditer(predicted_dose, flags=['multi_index'], op_flags=['writeonly'])
    x, y, z = fluence.shape
    n = 0
    while not it.finished:
        n += 1
        if not n % 10000:
            print(n)
        i, j, k = it.multi_index
        if i < INPUT_MARGIN or j < INPUT_MARGIN or k < INPUT_MARGIN:
            it.iternext()
            continue
        if x-i <= INPUT_MARGIN or y-j <= INPUT_MARGIN or z-k <= INPUT_MARGIN:
            it.iternext()
            continue
        x_in = [[[[
            float(i)/fluence.shape[0],
            float(j)/fluence.shape[1] - 0.5,
            float(k)/fluence.shape[2] - 0.5
            ]]]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_in[0][0][0] += [density[i+ii, j+jj, k+kk]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_in[0][0][0] += [integral[i+ii, j+jj, k+kk]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_in[0][0][0] += [fluence[i+ii, j+jj, k+kk]]

        x_in = np.array(x_in, dtype=PRECISION)
        ff = feed_forward(x_in)
        it[0] = ff[0]
        it.iternext()
    return predicted_dose

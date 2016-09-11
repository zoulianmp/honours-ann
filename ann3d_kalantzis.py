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

import numpy as np
import lasagne

def load_kalantzis_3d_dataset(dose, fluence):
    assert dose.shape == fluence.shape

    # The outer voxels of the dose tensors are unusable since the neighbouring
    # voxels are included as input to the neural network. We create a set of
    # all usable coordinates:
    coord_list = []
    for i in range(1, dose.shape[0]-1):
        for j in range(1, dose.shape[1]-1):
            for k in range(1, dose.shape[2]-1):
                coord_list.append([i,j,k])

    coord_list = np.array(coord_list)
    np.random.shuffle(coord_list)

    x_tmp = np.zeros((len(coord_list),1,1,10))
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
            fluence[c[0],c[1],c[2]+1],
            float(c[0])/fluence.shape[0],
            float(c[1])/fluence.shape[1] - 0.5,
            float(c[2])/fluence.shape[2] - 0.5
            ]
        y_tmp[i] = [dose[c[0],c[1],c[2]]]

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

    # (It doesn't matter how we do this as long as we can read them again.)
    return x_train, y_train, x_val, y_val, x_test, y_test

def kalantzis_3d(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 1 rows and 10 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 10), input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.0)

    # Add a fully-connected layer of 30 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=300,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 20%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

    # A 5-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 20% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.2)

    # Finally, we'll add the fully-connected output layer, of 1 rectifier unit:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=1,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

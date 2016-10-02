def ann3d_dataset(density, integral, fluence, dose):
    # The outer voxels of the dose tensors are unusable since the neighbouring
    # voxels are included as input to the neural network. We create a set of
    # susable coordinates:
    coord_list = sm.latin_hypercube(dose, INPUT_MARGIN, N_SAMPLES)

    print('coordinates sampled')

    n_inputs = 3+3*((2*INPUT_MARGIN+1)**3)
    x_list = np.empty((len(coord_list), 1, 1, n_inputs), dtype=np.float32)
    y_list = np.empty((len(coord_list), 1), dtype=np.float32)
    for i in range(len(coord_list)):
        c = coord_list[i]
        x_tmp = [
            float(c[1])/fluence[c[0]].shape[0],
            float(c[2])/fluence[c[0]].shape[1] - 0.5,
            float(c[3])/fluence[c[0]].shape[2] - 0.5
            ]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_tmp += [density[c[0]][c[1]+ ii, c[2]+jj, c[3]+kk]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_tmp += [integral[c[0]][c[1]+ ii, c[2]+jj, c[3]+kk]]
        for ii in range(-INPUT_MARGIN, INPUT_MARGIN+1):
            for jj in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                for kk in range(-INPUT_MARGIN, INPUT_MARGIN+1):
                    x_tmp += [fluence[c[0]][c[1]+ ii, c[2]+jj, c[3]+kk]]

        y_tmp = [dose[c[0]][c[1], c[2], c[3]]]

        x_list[i][0][0] = x_tmp
        y_list[i] = y_tmp

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

from __future__ import print_function
import os
import sys
import time
import datetime
import mhd_utils_3d as mhd
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

OLD_VOXEL_SIZE = (0.134,0.200,0.134)
NEW_VOXEL_SIZE = (0.125,0.125,0.125)

def main(file1, file2, file3):
    mhd1 = mhd.load_mhd(file1)[0]
    mhd2 = mhd.load_mhd(file2)[0]
    #mhd2 = zoom(mhd1,
    #        (OLD_VOXEL_SIZE[0]/NEW_VOXEL_SIZE[0],
    #         OLD_VOXEL_SIZE[1]/NEW_VOXEL_SIZE[1],
    #         OLD_VOXEL_SIZE[2]/NEW_VOXEL_SIZE[2]))
    #print(np.mean(mhd1))
    #for x in np.nditer(mhd1, op_flags=['readwrite']):
    #    if x < 0:
    #        x[...] = 0
    #    elif x < 1054.0:
    #        x[...] = (1.07/1054.0) * x
    #    elif x < 2819.0:
    #        x[...] = 1.07 + (2.17/2819.0) * (x-1054.0)
    #    else:
    #        x[...] = 2.17 + (8.00/9146.0) * (x-2819.0)
    mhd3 = mhd1 * mhd2
    mhd.write_mhd(file3, mhd3, mhd3.shape, NEW_VOXEL_SIZE)

if __name__ == '__main__':
    kwargs = {}
    kwargs['file1'] = sys.argv[1]
    kwargs['file2'] = sys.argv[2]
    kwargs['file3'] = sys.argv[3]
    main(**kwargs)

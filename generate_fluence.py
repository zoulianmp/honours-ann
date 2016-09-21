#!/usr/bin/env python
#coding=utf-8

#======================================================================
# Program:     Neural Network Dose Distribution
# Author:      James Keal
# Date:        2016/09/12
#======================================================================

'''
comment
'''

from __future__ import print_function

import sys
import datetime

import mhd_utils_3d as mhd
import numpy as np

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]
SS_DIST = 100.0                     # source-surface distance [cm]

MASS_ATT_AIR = 2.522e-2         # mass attenuation of air [cm^2/g]
MASS_ATT_WAT = 2.770e-2         # mass attenuation of water [cm^2/g]
DENSITY_AIR = 1.225e-3          # density of air [g/cm^3]
DENSITY_WAT = 1.000             # density of water [g/cm^3]

def print_progress(iteration, total):
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

def main(field_size, volume_file, output_file=None):
    if output_file is None:
        now = datetime.datetime.now()
        now = str(now.month) + '-' + str(now.day) + '_' + \
              str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        output_file = 'fluence_' + now + '.mhd'

    print_progress(0,3)
    vol = mhd.load_mhd(volume_file)[0]
    print_progress(1,3)
    fluence = generate_fluence(vol.shape, VOXEL_SIZE, (field_size,field_size))
    print_progress(2,3)
    mhd.write_mhd(output_file, fluence, fluence.shape, VOXEL_SIZE)
    print_progress(3,3)

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 3):
        print("Generates an MHD file containing the normalised fluence")
        print("defined by a given field size and volume.")
        print("Usage: %s <FIELD SIZE> <VOLUME> [OUTPUT FILE]" % sys.argv[0])
        print()
        print("FIELD SIZE: An integer defining the x-ray beam width at the")
        print("    surface of the phantom volume.")
        print("VOLUME: The path to an MHD file with volume equal to that")
        print("    in which the fluence will be generated")
        print("OUTPUT FILE: The name of the MHD file that will contain the")
        print("    generated data.")
    else:
        kwargs = {}
        kwargs['field_size'] = int(sys.argv[1])
        kwargs['volume_file'] = sys.argv[2]
        if len(sys.argv) > 3:
            kwargs['output_file'] = sys.argv[3]
        main(**kwargs)

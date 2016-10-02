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

def main(density_file, output_file=None):
    if output_file is None:
        now = datetime.datetime.now()
        now = str(now.month) + '-' + str(now.day) + '_' + \
              str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        output_file = 'integral_' + now + '.mhd'

    print_progress(0,3)
    density = mhd.load_mhd(density_file)[0]
    print_progress(1,3)
    density = np.cumsum(density[::-1,...]-DENSITY_WAT, axis=0)[::-1,...]/40.0
    print_progress(2,3)
    mhd.write_mhd(output_file, density, density.shape, VOXEL_SIZE)
    print_progress(3,3)

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 2):
        print("Generates an MHD file containing the integral of voxels")
        print("defined by a given density distribution.")
        print("Usage: %s <DENSITY> [OUTPUT_FILE]" % sys.argv[0])
        print()
        print("DENSITY: The path to an MHD file with containing density data")
        print("    to be converted into an integral volume.")
        print("OUTPUT_FILE: The name of the MHD file that will contain the")
        print("    integral data.")
    else:
        kwargs = {}
        kwargs['density_file'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['output_file'] = sys.argv[2]
        main(**kwargs)

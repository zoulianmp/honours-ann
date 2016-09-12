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

import mhd_utils_3d as mhd
import numpy as np

VOXEL_SIZE = (0.125,0.125,0.125)

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

def import_dose_from_gate(root_dir, unit='dose'):
    if unit == 'dose':
        unit = '/output-Dose.mhd'
    elif unit == 'energy':
        unit = '/output-Edep.mhd'
    else:
        print("Unrecognized measurement type %r." % unit)
        return

    dirs = os.walk(root_dir).next()[1]
    n_dir = 0
    print_progress(n_dir, len(dirs))
    for path in dirs:
        n_dir += 1
        print_progress(n_dir, len(dirs))
        ddata = mhd.load_mhd(root_dir + '/' + path + unit)[0]
        if n_dir == 1:
            data = ddata
        else:
            data += ddata

    return data

def main(source_dir, unit='dose', output_file='combined.mhd'):
        dose = import_dose_from_gate(source_dir, unit)
        dose = dose/np.max(dose)
        mhd.write_mhd(output_file, dose, dose.shape, VOXEL_SIZE)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or (len(sys.argv) < 2):
        print("Combines MHD files produced by GATE into one normalised file.")
        print("Usage: %s <SOURCE DIRECTORY> [UNIT] [OUTPUT FILE]" % sys.argv[0])
        print()
        print("SOURCE DIRECTORY: The folder containing subfolders that each")
        print("    contain the output of one linac simulation.")
        print("UNIT: The type of data to be combined. Options are:")
        print("    'dose' for volumetric dose")
        print("    'energy' for energy deposited")
        print("    (default: dose)")
        print("OUTPUT FILE: The name of the MHD file that will contain the")
        print("    combined data.")
    else:
        kwargs = {}
        kwargs['source_dir'] = sys.argv[1]
        kwargs['unit'] = sys.argv[2]
        if len(sys.argv) > 3:
            kwargs['output_file'] = sys.argv[3]
        main(**kwargs)

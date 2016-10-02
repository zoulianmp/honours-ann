#!/usr/bin/env python
#coding=utf-8

#======================================================================
# Program:     Neural Network Dose Distribution
# Author:      James Keal
# Date:        2016/09/25
#======================================================================

'''
comment
'''

from __future__ import print_function
from re import sub

import os
import sys
import datetime
import dicom as dc
import mhd_utils_3d as mhd
import numpy as np
import matplotlib.pyplot as plt

VOXEL_SIZE = (0.125,0.125,0.125)    # voxel size (z,y,x) [cm]

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

def main(dicom_dir, output_file=None, dims=2):
    if output_file is None:
        now = datetime.datetime.now()
        now = str(now.month) + '-' + str(now.day) + '_' + \
              str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        output_file = 'dicom_' + now + '.mhd'

    if dims == 2:
        print_progress(0,3)
        file_dict = {}
        for dirpath, subdirs, files in os.walk(dicom_dir):
            for f in files:
                if f.endswith(".dcm"):
                    file_dict[int(sub(r'[^0-9]*',"",f))] = os.path.join(dirpath,f)

        # Natural sort by number in filename
        print_progress(1,3)
        dicom = []
        for f in sorted(file_dict):
            dicom += [dc.read_file(file_dict[f]).pixel_array]
        dicom = np.array(dicom)

        dicom = np.moveaxis(dicom, 0, 1)
        dicom = np.flipud(dicom)

        print_progress(2,3)
        mhd.write_mhd(output_file, dicom, dicom.shape, VOXEL_SIZE)
        print_progress(3,3)

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 2):
        print("Generates an MHD file containing the density of voxels")
        print("defined by the slices of a DICOM CT scan.")
        print("Usage: %s <DICOM> [OUTPUT_FILE]" % sys.argv[0])
        print()
        print("VOLUME: The path to a directory containing DCM files, each")
        print("    representing a layer of the CT scanned volume.")
        print("OUTPUT_FILE: The name of the MHD file that will contain the")
        print("    generated data.")
    else:
        kwargs = {}
        args = sys.argv
        if ('--3d' in args):
            kwargs['dims'] = 3
            args = [a for a in args if a != '--3d']
        else:
            kwargs['dims'] = 2
        kwargs['dicom_dir'] = args[1]
        if len(args) > 2:
            kwargs['output_file'] = args[2]
        main(**kwargs)

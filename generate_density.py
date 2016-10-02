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

DENSITY_ADI = 0.920             # density of adipose [g/cm^3]
DENSITY_AIR = 1.225e-3          # density of air [g/cm^3]
DENSITY_RIB = 1.920             # density of rib bone [g/cm^3]
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

def generate_density(vol_shape, vox_size, geometry, dens1, dens2=None):
    if geometry == 'homo':
        density = dens1*np.ones(vol_shape)
        return density

    elif geometry == 'slabs':
        slab_size = 5.0         # slab thickness [cm]

        n_slab_vox = int(slab_size/vox_size[0])
        if 2*n_slab_vox <= vol_shape[0]:
            n_water_vox = int(vol_shape[0]/2.0 - n_slab_vox)
            water = DENSITY_WAT*np.ones((n_water_vox,vol_shape[1],vol_shape[2]))
            mat1 = dens1*np.ones((n_slab_vox,vol_shape[1],vol_shape[2]))
            mat2 = dens2*np.ones((n_slab_vox,vol_shape[1],vol_shape[2]))
            density = np.concatenate((water, mat2, mat1, water))
        else:
            mat1 = dens1*np.ones((vol_shape[0]/2.0,vol_shape[1],vol_shape[2]))
            mat2 = dens2*np.ones((vol_shape[0]/2.0,vol_shape[1],vol_shape[2]))
            density = np.concatenate((mat2, mat1))
        return density

    elif geometry == 'cubes':
        cube_one = 3.0          # cube one size [cm]
        cube_two = 1.0          # cube two size [cm]
        cube_rad = 2.5          # cube displacement [cm]

        cube_one = int(cube_one/vox_size[0])
        cube_rad = int(cube_rad/vox_size[0])
        cube1 = dens1*np.ones((cube_one,cube_one,cube_one))
        cube1 = np.lib.pad(cube1, ((int((vol_shape[0]-cube_one)/2.0),
                                    int((vol_shape[0]-cube_one)/2.0)),
                                   (int((vol_shape[1]-cube_one)/2.0),
                                    int((vol_shape[1]-cube_one)/2.0)),
                                   (int(cube_rad-cube_one/2.0),
                                    int((vol_shape[2]-cube_one)/2.0-cube_rad))),
                'constant', constant_values=(DENSITY_WAT,))
        cube_two = int(cube_two/vox_size[0])
        cube2 = dens1*np.ones((cube_two,cube_two,cube_two))
        cube2 = np.lib.pad(cube2, ((int((vol_shape[0]-cube_two)/2.0),
                                    int((vol_shape[0]-cube_two)/2.0)),
                                   (int((vol_shape[1]-cube_two)/2.0),
                                    int((vol_shape[1]-cube_two)/2.0)),
                                   (int((vol_shape[2]-cube_two)/2.0-cube_rad),
                                    int(cube_rad-cube_two/2.0))),
                'constant', constant_values=(DENSITY_WAT,))
        density = np.concatenate((cube2, cube1), axis=2)
        return density

    else:
        print("Unrecognized geometry type %r." % geometry)
        return

def main(geometry, integral, volume_file, mat1, mat2='none', output_file=None):
    if output_file is None:
        now = datetime.datetime.now()
        now = str(now.month) + '-' + str(now.day) + '_' + \
              str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        if integral:
            output_file = 'integral_' + now + '.mhd'
        else:
            output_file = 'density_' + now + '.mhd'

    print_progress(0,3)
    meta = mhd.read_meta_header(volume_file)
    vol = [int(i) for i in meta['DimSize'].split()]
    vol.reverse()

    dens = {'adipose': DENSITY_ADI,
            'air':     DENSITY_AIR,
            'ribbone': DENSITY_RIB,
            'water':   DENSITY_WAT,
            'none':    None}

    print_progress(1,3)
    density = generate_density(vol, VOXEL_SIZE, geometry, dens[mat1], dens[mat2])
    if integral:
        density = np.cumsum(density[::-1,...]-DENSITY_WAT, axis=0)[::-1,...]/40.0
    print_progress(2,3)
    mhd.write_mhd(output_file, density, density.shape, VOXEL_SIZE)
    print_progress(3,3)

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 4):
        print("Generates an MHD file containing the density of voxels")
        print("defined by a given geometry and set of materials.")
        print("Usage: %s [FLAGS] <GEOMETRY> <VOLUME> <MAT1> [[MAT2] OUTPUT_FILE]"
                % sys.argv[0])
        print()
        print("FLAGS:")
        print("--integral: Setting this flag displays the plots in a window.")
        print()
        print("GEOMETRY: A string defining the type of volumetric arrangement")
        print("    to be returned. Options are:")
        print("    'homo' for a volume of uniform density.")
        print("    'slabs' for a pair of 50mm slabs perpendicular to the beam.")
        print("    'cubes' for a pair of cubes, 10mm and 30mm in size.")
        print("VOLUME: The path to an MHD file with volume equal to that")
        print("    in which the density will be generated")
        print("MAT1: The material of the first inhomogeneity. Options are:")
        print("    'adipose', 'air', 'ribbone', or 'water'")
        print("MAT1: The material of the second inhomogeneity. Options are:")
        print("    'adipose', 'air', 'ribbone', 'water', or 'none'")
        print("OUTPUT_FILE: The name of the MHD file that will contain the")
        print("    generated data.")
    else:
        kwargs = {}
        args = sys.argv
        if ('--integral' in args):
            kwargs['integral'] = True
            args = [a for a in args if a != '--integral']
        else:
            kwargs['integral'] = False
        kwargs['geometry'] = args[1]
        kwargs['volume_file'] = args[2]
        kwargs['mat1'] = args[3]
        if len(args) > 4:
            kwargs['mat2'] = args[4]
        if len(args) > 5:
            kwargs['output_file'] = args[5]
        main(**kwargs)

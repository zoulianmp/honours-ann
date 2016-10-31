from __future__ import print_function
import sys
import mhd_utils_3d as mhd
import numpy as np

def main(mhd_file):
    [data, meta] = mhd.load_mhd(mhd_file)
    print("MHD INFO")
    print("  Dimensions:\t\t%s" % meta['DimSize'])
    print("  Voxel Size:\t\t%s" % meta['ElementSpacing'])
    print("  Max value:\t\t%s" % np.max(data))
    print("  Min value:\t\t%s" % np.min(data))
    print("  Mean value:\t\t%s" % np.mean(data))

if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv) < 2):
        print("Provides info about an MHD file.")
        print("Usage: %s <MHD_FILE>" % sys.argv[0])
    else:
        kwargs = {}
        kwargs['mhd_file'] = sys.argv[1]
        main(**kwargs)

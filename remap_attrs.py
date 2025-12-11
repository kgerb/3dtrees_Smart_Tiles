import numpy as np
from scipy.spatial import KDTree

def remap(ins, outs):
    # Extract merged cloud
    data = ins

    # Determine mask for original vs subsampled
    # We assume original points come first (from merge order)
    # Count how many points in original
    n_orig = outs['X'].shape[0]
    n_total = data['X'].shape[0]
    n_sub = n_total - n_orig

    # separate coords
    orig_xyz = np.vstack([outs['X'], outs['Y'], outs['Z']]).T
    sub_xyz  = np.vstack([data['X'][n_orig:], data['Y'][n_orig:], data['Z'][n_orig:]]).T

    # build kd-tree on subsampled points
    tree = KDTree(sub_xyz)
    dist, idx = tree.query(orig_xyz)

    # map attribute from subsampled cloud
    # modify here for other attributes
    src_attr = data['PredInstance'][n_orig:]
    outs['PredInstance'] = src_attr[idx]

    return True

import numpy as np

def load_calib(path):
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in val.strip().split()])

    if 'R' in calib and 'T' in calib:
        R = calib['R'].reshape(3, 3)
        T = calib['T'].reshape(3, 1)
        Tr = np.hstack((R, T))  # Create 3x4 matrix
        calib['Tr_velo_to_cam'] = Tr

    return calib

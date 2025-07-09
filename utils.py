import numpy as np

def load_calib(path):
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line: continue
            key, val = line.split(':', 1)
            calib[key] = np.array([float(x) for x in val.strip().split()])
    return calib
import numpy as np
import cv2
from utils import load_calib

def get_rgbd(image_path, lidar_path, calib_path, output_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    calib = load_calib(calib_path)
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]

    P = calib['P2'].reshape(3,4)
    Tr = calib['Tr_velo_to_cam'].reshape(3,4)
    pts_hom = np.hstack((lidar, np.ones((lidar.shape[0],1))))
    proj = (P @ Tr @ pts_hom.T).T
    proj = proj[:, :2] / proj[:, 2:3]

    depth_map = np.zeros(img.shape[:2], dtype=np.float32)
    for (u, v), d in zip(proj.astype(int), lidar[:, 2]):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            depth_map[v, u] = d

    img = cv2.resize(img, output_size)
    depth_map = cv2.resize(depth_map, output_size)
    rgbd = np.dstack((img, depth_map)).astype(np.float32) / 255.0
    return rgbd

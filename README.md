# Camera–LiDAR Fusion for Scene Classification using KITTI

This project demonstrates how to fuse RGB camera images and 3D LiDAR point cloud data to classify driving scenes using deep learning. It uses data from the KITTI dataset and a modified ResNet18 model to accept 4-channel input (RGB + projected depth from LiDAR).

---

## Overview

Most vision models use images alone — but autonomous vehicles rely on both **camera** and **LiDAR** sensors. This project:

- Parses KITTI calibration files
- Projects 3D LiDAR points into the camera frame
- Creates a **depth map** from LiDAR
- Combines RGB and depth into a 4-channel image
- Feeds the fused input into a modified ResNet18 model for classification

---

## Project Structure

camera_lidar_fusion_dl/
├── data/
│ ├── 000000.png # KITTI camera image
│ ├── 000000.bin # KITTI LiDAR point cloud
│ └── calib.txt # Calibration parameters
├── preprocess.py # Projects LiDAR to image plane, creates RGB-D
├── model.py # Modified 4-channel ResNet
├── inference.py # Loads model and runs prediction
├── utils.py # Calibration file parser
└── README.md

## Run

```bash
pip install torch cv2 torchvision 
python inference.py
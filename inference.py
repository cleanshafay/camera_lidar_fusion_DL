import torch
import torchvision.transforms as T
from model import RGBDResNet
from preprocess import get_rgbd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

rgbd = get_rgbd("data/000000.png", "data/000000.bin", "data/calib.txt")
transform = T.ToTensor()
input_tensor = transform(rgbd).unsqueeze(0).to(device)  # [1, 4, 224, 224]

model = RGBDResNet(num_classes=3).to(device)
model.eval()

with torch.no_grad():
    logits = model(input_tensor)
    pred = torch.argmax(logits, dim=1).item()

labels = ['Empty Road', 'Car Ahead', 'Pedestrian']
print(f"Predicted: {labels[pred]}")

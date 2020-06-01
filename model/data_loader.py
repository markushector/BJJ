import numpy as np
from PIL import Image
import cv2
import torch
import os
import torchvision
from torch.utils.data import DataLoader, Dataset

imagenet_data = torchvision.datasets.ImageNet('../data/data')
data_loader = DataLoader(imagenet_data, batch_size=10, shuffle=True)

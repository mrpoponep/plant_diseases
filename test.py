import torch
from PIL import Image
import os                       # for working with files
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision import datasets,transforms,models   # for working with classes and images
import pandas as pd
import torch.nn.functional as nnf
from config import class_names

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
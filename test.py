from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import torch
from tqdm.auto import tqdm
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision import datasets,transforms,models   # for working with classes and images
from tempfile import TemporaryDirectory           # for getting the summary of our model
from tqdm import tqdm
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from pycm import ConfusionMatrix
from config import class_names
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 63)
model_ft = model_ft.to("cuda")

batch_size = 16
data_transforms = {
    'Train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'C:/code/pytorch_course/plant_disease/Data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Train', 'Test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=2)
              for x in ['Train', 'Test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}
class_names = image_datasets['Train'].classes


model_ft.load_state_dict(torch.load('Model/resnet18_final.pt'))
model_ft.eval()
torch.save(model_ft, 'Model/resnet18_full_2.pth')
# 1. Make predictions with trained model
def making_pred():    
    y_true=[]
    y_preds = []
    model_ft.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloaders["Test"], desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to("cuda"), y.to("cuda")
            # Do the forward pass
            y_logit = model_ft(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
            y_true.append(y)
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_true)
    y_pred_tensor=y_pred_tensor.cpu().numpy()
    y_true_tensor=y_true_tensor.cpu().numpy()
    cm = ConfusionMatrix(actual_vector=y_true_tensor, predict_vector=y_pred_tensor,classes=class_names)
    cm.save_html("confusion.html", color=(255, 0, 0))
    cm.save_csv("confusion.cvs")
    

if __name__ == '__main__':
    pass


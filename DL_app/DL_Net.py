import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def CNN3d(nn.Module):
    def __init__(self, Layers):
        super(CNN3d, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                                          nn.Conv2d(1, 8, kernel_size=7),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(8, 10, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True)
                                          )
            
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
                                    nn.Linear(10 * 3 * 3, 32),
                                    nn.ReLU(True),
                                    nn.Linear(32, 3 * 2)
                                    )
                                          
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Calculate the range of values for uniform distributions
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
               
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        ## Output size = (W-F + 2*P)/S + 1 = (224-3 + 2*1)/1 + 1 = 224
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1) # -> (32, 224, 224)       
        # Maxpool layer; pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2) # -> (32, 112, 112)
        ## Output size = (W-F + 2*P)/S + 1 = (112-3 + 2*1)/1 + 1 = 112
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1) # -> (64, 112, 112)
        self.pool2 = nn.MaxPool2d(2, 2) # -> (64, 56, 56)
        ## Output size = (W-F + 2*P)/S + 1 = (56-3 + 2*1)/1 + 1 = 224
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1) # -> (128, 56, 56)
        self.pool3 = nn.MaxPool2d(2, 2) # -> (128, 28, 28)
        self.do1 = nn.Dropout(0.4)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.do2 = nn.Dropout(0.4)
        # 136 output values, 2 for each of the 68 keypoint (x, y) pairs
        self.fc2 = nn.Linear(512, 136)
        
        self.reset_parameters()
       
    def reset_parameters(self):
        # Apply to layers the specified weight initialization
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
          
    def forward(self, x):
        # Three conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.do1(x)
     
        # Prep for linear layer
        # Flatten the inputs into a vector
        x = x.view(x.size(0), -1)
   
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        x = self.fc2(x)
        return x
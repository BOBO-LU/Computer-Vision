
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        # print("LeNet x.shape: ", x.shape)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class residual_block(nn.Module):
    def __init__(self):
        super(residual_block, self).__init__()
        self.relu = nn.ReLU()
        self.blocks = nn.Identity()
        self.activate = nn.ReLU()
        self.shortcut = nn.Identity()   

    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        residual = x
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
        
        pass

class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self, in_size:int, hidden_size:int, out_size:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
   
    """
    Combine output with the original input
    """
    def forward(self, x): return x + self.convblock(x) # skip connection
        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.residual1 = ResBlock(16, 16, 16)
        self.residual2 = ResBlock(16, 16, 16)
        # self.residual2 = residual_block()
        # self.residual3 = residual_block()

        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_out)

        pass
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)


        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual1(x)
        # print("ResNet x.shape: ", x.shape)
        x = self.residual2(x)
        # print("ResNet x.shape: ", x.shape)
        # x = self.residual3(x)
        # print("ResNet x.shape: ", x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        # print("ResNet x.shape: ", x.shape)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

        pass
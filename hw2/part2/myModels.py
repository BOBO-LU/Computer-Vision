
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  

    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(x)
        x = F.relu(x)
        return x
        

class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        self.conv1 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.BatchNorm2d(64),
                             nn.ReLU())

        self.residual1 = residual_block(64, 64, stride=1)

        self.conv2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.BatchNorm2d(128),
                             nn.ReLU(),)

        self.residual2 = residual_block(128, 128, stride=1)

        self.conv3 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.BatchNorm2d(256),
                             nn.ReLU(),)

        self.residual3 = residual_block(256, 256, stride=1)

        self.residual4 = residual_block(256, 256, stride=1)


        self.fc1 = nn.Sequential(nn.Linear(1024, 240), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(240, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_out)

        pass
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)

        x = self.stem_conv(x)
        x = self.conv1(x)
        x = self.residual1(x)

        x = self.conv2(x)
        x = self.residual2(x)

        x = self.conv3(x)
        x = self.residual3(x)

        x = self.residual4(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        # print("ResNet x.shape: ", x.shape)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

def pretrained_ResNet50(num_out):
    model = torchvision.models.resnet50(pretrained=True)
    num_fc_ftr = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_ftr, num_out)
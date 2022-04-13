
import torch
import os


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from myModels import  myLeNet, myResnet, pretrained_ResNet50
from dla import DLA 
from myDatasets import  get_cifar10_train_val_set
from tool import train, fixed_seed
import torchvision.models as models

# Modify config if you are conducting different models
# from cfg import LeNet_cfg as cfg
# from cfg import ResNet_cfg as cfg
from cfg import DLA_cfg as cfg 
# from cfg import preResNet_cfg as cfg
import argparse

def train_interface():
    
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cleanning', help='cleanning or not', type=bool, default=False)
    # args = parser.parse_args()
    
    # cleanning = args.cleanning

    """ input argumnet """
    cleanning = cfg['cleanning'] # clean data or not
    semi = cfg['semi'] # add semi data or not
    if semi:
        semi_root = cfg['semi_root']
    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']

    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    # model = myLeNet(num_out=num_out)
    # model = myResnet(num_out=num_out)
    model = DLA(num_classes=num_out)
    # model = pretrained_ResNet50(num_out=num_out)
    
    

    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio, cleanning=cleanning, semi_root=semi_root)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py

    train(model=model, train_loader=train_loader, val_loader=val_loader, 
            num_epoch=num_epoch, log_path=log_path, save_path=save_path,
            device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    
if __name__ == '__main__':
    train_interface()




    
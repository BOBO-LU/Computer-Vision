

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet', # origin : ResNet
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 3, # origin : 30
    'cleanning' : False
    
}

ResNet_cfg = {
    'model_type': 'ResNet', # origin : ResNet
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 50, # origin : 30
    'cleanning' : False
    
}

DLA_cfg = {
    'model_type': 'DLA', # origin : ResNet
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 50, # origin : 30
    'cleanning' : False
    
}
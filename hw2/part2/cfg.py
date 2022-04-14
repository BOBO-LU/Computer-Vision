

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
    'num_epoch': 30, # origin : 30
    'cleanning' : False,
    'semi': False,
    'semi_root': ''
    
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
    'num_epoch': 30, # origin : 30
    'cleanning' : False,
    'semi': False,
    'semi_root': ''
    
}

DLA_cfg = {
    'model_type': 'DLA', # origin : ResNet
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 64,
    'lr':0.01,
    'milestones': [15, 20, 25],
    'num_out': 10,
    'num_epoch': 30, # origin : 30
    'cleanning' : True,
    'semi': True,
    'semi_root': './p2_data/annotations/semi_annos.json'
    
}

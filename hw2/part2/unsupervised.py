
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

from tool import load_parameters
from myModels import myResnet, myLeNet
from dla import DLA 
from myDatasets import cifar10_dataset



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='model_path', type=str, default='')
    parser.add_argument('--data_root', help='annotaion for test image', type=str, default= './p2_data/annotations/train_annos.json')
    args = parser.parse_args()

    path = args.path
    data_root = args.data_root
    prefix = './p2_data/train'
    # change your model here
    
    ## TO DO ## 
    # Indicate the model you use here
    # model = myLeNet(num_out=10) 
    # model = myResnet(num_out=10) 
    model = DLA(num_classes=10) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    # Simply load parameters
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location='cuda')
    model.load_state_dict(param)
    print("End of loading !!!")
    model.to(device)


    with open(data_root, 'r') as f :
        data = json.load(f)    
    
    imgs, categories = data['images'], data['categories']
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    train_set = cifar10_dataset(images=imgs, labels= categories, transform=trans, prefix = prefix)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for img, label in tqdm(train_loader):
            
            img = img.to(device)
            pred = model(img)
            
            
                

    print(len(pred_list))
    small_index = np.argpartition(pred_list, 3000)[:3000]
    print(small_index)

    import csv 
    with open('clean.csv', 'w+') as f:
        write = csv.writer(f) 
        write.writerow(small_index) 

    
    
if __name__ == '__main__':
    main()
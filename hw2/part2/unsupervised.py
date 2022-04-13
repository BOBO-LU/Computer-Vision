
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm
import numpy as np
import os
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
    semisuper_data_root = './p2_data/annotations/semi_annos.json'
    prefix = './p2_data/unlabeled'
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

    # get all filename from folder
    unlabel_imgs = next(os.walk(prefix), (None, None, []))[2]
    unlabel_imgs = [f for f in unlabel_imgs if f.find('.jpg') != -1]

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    train_set = cifar10_dataset(images=unlabel_imgs, labels=None, transform=trans, prefix = prefix)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    
    model.eval()
    label_list = []
    with torch.no_grad():
        for img, _ in tqdm(train_loader):
            
            img = img.to(device)
            pred = model(img)

            for batch_idx in range(pred.shape[0]):

                # if max res is less than threshold, discard (label = -1)

                if torch.max(pred[batch_idx]) < 8:
                    label_list.append(-1)
                else:
                    label_list.append(int(torch.argmax(pred[batch_idx])))



            
    print(label_list[:20])
    print(len(label_list))
    

    new_imgs = []
    new_labels = []
    for img_name, pred_label in zip(unlabel_imgs, label_list):
        if pred_label != -1:
            new_imgs.append(img_name)
            new_labels.append(pred_label)

    data = {'images':new_imgs, 'categories':new_labels}
    with open(semisuper_data_root, 'w+') as f :
        json.dump(data, f)    
    


    

    # import csv 
    # with open('clean.csv', 'w+') as f:
    #     write = csv.writer(f) 
    #     write.writerow(label_list) 

    
    
if __name__ == '__main__':
    main()

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



# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def test_result(test_loader, model, device):
    pred = []
    cnt = 0
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            print("pred: ", pred)
            break
            raise 'e'
    acc = cnt / len(test_loader.dataset)
    return acc

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
    load_parameters(model=model, path=path)
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
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for img, label in tqdm(train_loader):
            
            img = img.to(device)
            pred = model(img)
            
            for batch_img, batch_label in zip(range(pred.shape[0]), label):
                pred_list.append(float(pred[batch_img][batch_label]))

    print(len(pred_list))
    small_index = np.argpartition(pred_list, 3000)[:3000]
    print(small_index)

    import csv 
    with open('clean.csv', 'w+') as f:
        write = csv.writer(f) 
        write.writerow(small_index) 

    
    
if __name__ == '__main__':
    main()
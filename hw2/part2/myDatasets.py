

from email.mime import image
import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 

from torchvision.transforms import transforms
from PIL import Image
import json 



def get_cifar10_train_val_set(root, ratio=0.9, cv=0, cleanning=False):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']
    
    # clean img
    if cleanning:
        print("Train with clean data !!!")
        new_images, new_labels = [], []
        dirty_index_list = []
        in_filename = "clean.csv"
        with open(in_filename) as in_file:
            for line in in_file:
                small_list = [int(x) for x in line.split(',')]
            for s in small_list:
                dirty_index_list.append(images[s])
                
        for img, lab in zip(images, labels):
            if img in dirty_index_list:
                continue
            # if '99' in img:
            # continue
            new_images.append(img)
            new_labels.append(lab)

        images, labels = new_images, new_labels

    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio)
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]
    

    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                ## TO DO ##
                # You can add some transforms here
                transforms.RandomVerticalFlip(0.3), 
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.1),

                # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
                transforms.ToTensor(),
                
                # experimental normalization for image classification 
                transforms.Normalize(means, stds),

                transforms.RandomErasing(0.3),
            ])
  
    # normally, we dont apply transform to test_set or val_set
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

 
  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform)


    return train_set, val_set



## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        
        img_path = os.path.join(self.prefix, self.images[idx])


        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, label

        # You shall return image, label with type "long tensor" if it's training set
        pass
        

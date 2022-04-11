from email.mime import image
from PIL import Image
import numpy as np
import operator

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.size, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def center_crop(img):        
    min_length = min(img.size)
    left = int(img.size[0]/2-min_length/2)
    upper = int(img.size[1]/2-min_length/2)
    right = left + min_length
    lower = upper + min_length

    img_cropped = img.crop((left, upper,right,lower))
    return img_cropped
    
def get_tiny_images(image_paths, zero_mean=False, target_size=(16,16)):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    
    
    TARGET_SIZE = target_size
    zero_mean=True
    '''
    ALGO:
    1. read img 
    2. crop the center square
    3. resize
    4. flatten
    5. normalize
    '''
    tiny_images = []

    for img_path in image_paths:
        img = Image.open(img_path)
        crop_img = center_crop(img)
        resize_img = crop_img.resize(TARGET_SIZE)
        flatten_img = np.array(resize_img).flatten()

        if zero_mean:
            flatten_img = (flatten_img - flatten_img.mean(axis=0)) / flatten_img.std(axis=0)
        
        normalize_img = flatten_img / np.linalg.norm(flatten_img)

        tiny_images.append(normalize_img)
    
    return tiny_images

    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

if __name__ == '__main__':
    get_tiny_images()

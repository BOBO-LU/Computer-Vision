from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=5):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input :
        train_image_feats :
            image_feats is an (N, d) matrix, where d is the
           dimensionality of the feature representation.

        train_labels :
            image_feats is a list of string, each string
            indicate the ground truth category for each training image.

        test_image_feats :
            image_feats is an (M, d) matrix, where d is the
            dimensionality of the feature representation.
    Output :
        test_predicts :
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''

    '''
    ALGO
    1. calculate all distance
    2. iterate each test image
    3. sort distance
    4. find k smallest images
    5. vote label
    '''
    # LABELS = ['Forest', 'Bedroom', 'Office', 'Highway', 'Coast', 'InsideCity', 'TallBuilding', 'Industrial', 'Street',
    #           'LivingRoom', 'Suburb', 'Mountain', 'Kitchen', 'OpenCountry', 'Store']

    test_predicts = []

    distance_list = distance.cdist(train_image_feats, test_image_feats)
    train_labels = np.array(train_labels)

    for test_idx, _ in enumerate(test_image_feats):
        distances = distance_list[:, test_idx]
        min_index = np.argpartition(distances, k)[:k]
        top_k_features = train_labels[min_index]
        unique, pos = np.unique(top_k_features, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        label = unique[maxpos]

        test_predicts.append(label)

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts

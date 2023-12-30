import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import torch

import torchvision
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

from numpy import expand_dims, moveaxis
from numpy import asarray
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def data_prep_pytorch_mnist(batch_size, image_size):

    
    
    data_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()])
    # Setup training data
    train_data = datasets.FashionMNIST(root="mnist_data", # where to download data to?
                                       train=True, # get training data
                                       download=True, # download data if it doesn't exist on disk
                                       transform=data_transform, # images come as PIL format, we want to turn into Torch tensors
                                       target_transform=None # you can transform labels as well
                                      )

    # Setup testing data
    test_data = datasets.FashionMNIST(root="mnist_data",
                                      train=False, # get test data
                                      download=True,
                                      transform=data_transform)


    
    
    
    
    # Data Loader

    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data, # dataset to turn into iterable
                                  batch_size=batch_size, # how many samples per batch? 
                                  shuffle=True # shuffle data every epoch?
                                 )

    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False
                                )

    return train_dataloader, test_dataloader, train_data.targets, test_data.targets, train_data.classes




def data_prep_archive(batch_size, image_size, data_transform):


    # 1. Create multi-class data
    # Train
    train_directory ="archive_/train/"
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(train_directory)
    
    # shuffle the dataset
    control = 'label'
    random_order = np.random.permutation(len(train))
    train['RandomOrder'] = random_order
    train = train.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    train = train.drop(columns=['RandomOrder'])
    
    
    # Test
    test_directory = "archive_/test/"
    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(test_directory)
    
    # shuffle the dataset
    control = 'label'
    random_order = np.random.permutation(len(test))
    test['RandomOrder'] = random_order
    test = test.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    test = test.drop(columns=['RandomOrder'])
    
    
    
    val_directory = "archive_/validation/"
    val = pd.DataFrame()
    val['image'], val['label'] = load_dataset(val_directory)

    # shuffle the dataset
    control = 'label'
    random_order = np.random.permutation(len(val))
    val['RandomOrder'] = random_order
    val = val.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    val = val.drop(columns=['RandomOrder'])
    
    
    
    
    train_features = extract_features(train['image'], image_size, data_transform)
    test_features = extract_features(test['image'], image_size, data_transform)
    val_features = extract_features(val['image'], image_size, data_transform) 
    

    

    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    y_val = le.transform(val['label'])
    
    
    
    # 2. Turn data into tensors using data loaders
    # Data Loader
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(list(zip(train_features, y_train)),
                                  batch_size=batch_size, # how many samples per batch? 
                                  shuffle=True # shuffle data every epoch?
                                 )

    
    test_dataloader = DataLoader(list(zip(test_features, y_test)),
                                 batch_size=batch_size,
                                 shuffle=True)
    
    val_dataloader = DataLoader(list(zip(val_features, y_val)), 
                                batch_size=32, shuffle=False)
    
    classes = ['Bird','Cat', 'Dog']
    return train_dataloader, test_dataloader, val_dataloader, torch.from_numpy(y_train), torch.from_numpy(y_test),  torch.from_numpy(y_val), classes






def load_dataset(directory):
    image_paths = []
    labels = []
    
    directory_ = os.listdir(directory)
    
    for i in directory_:
        #path = os.listdir(directory + folder)
        if i.startswith('.') or i.startswith('_'):
            directory_.remove(i)
        
    for folder in directory_:
        for filename in os.listdir(directory+folder):
            image_path = os.path.join(directory, folder, filename)

            if filename.startswith('.') or filename.startswith('_') or filename[-4:] != '.jpg':
                pass

            else:
                image_paths.append(image_path)
                labels.append(folder)
        
    return image_paths, labels



def extract_features(images, image_size, data_transform):
    
    features = []
    for image in tqdm(images):
        img = Image.open(image)
        img = img.convert('RGB')
        data = data_transform(img)
        features.append(data)
    return features
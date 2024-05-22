import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.utils import to_categorical



def data_prep_archive(batch_size, image_size):

    # 1. Create multi-class data
    # Train
    train_directory ="archive/train/"
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(train_directory)
    
    # shuffle the dataset
    control = 'label'
    random_order = np.random.permutation(len(train))
    train['RandomOrder'] = random_order
    train = train.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    train = train.drop(columns=['RandomOrder'])
    
    
    # Test
    test_directory = "archive/test/"
    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(test_directory)
    
    # shuffle the dataset
    control = 'label'
    random_order = np.random.permutation(len(test))
    test['RandomOrder'] = random_order
    test = test.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    test = test.drop(columns=['RandomOrder'])
    
    
    
    val_directory = "archive/validation/"
    val = pd.DataFrame()
    val['image'], val['label'] = load_dataset(val_directory)

    # shuffle the dataset
    control = 'label'
    random_order = np.random.permutation(len(val))
    val['RandomOrder'] = random_order
    val = val.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    val = val.drop(columns=['RandomOrder'])
    
    
    train_features = extract_features(train['image'], image_size)
    test_features = extract_features(test['image'], image_size)
    val_features = extract_features(val['image'], image_size) 


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



def extract_features(images, image_size):
    data_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()])
    
    
    features = []
    for image in tqdm(images):
        img = Image.open(image)
        img = img.convert('RGB')
        data = data_transform(img)
        features.append(data)
    return features
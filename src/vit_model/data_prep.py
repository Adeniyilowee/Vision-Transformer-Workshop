import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder


def data_prep_archive(batch_size, image_size, data_type):
    """
    Prepare data for training, validation, and testing.

    Args:
        batch_size (int): Number of samples per batch.
        image_size (int): Size of the input images.
        data_type (str): Type of data files ('jpg' or 'tiff').

    Returns:
        tuple: Tuple containing train, test, and validation dataloaders, label encodings, and classes.

    """
    # 1. Create multi-class data
    # Train
    train_directory = "src/vit_model/archive/train/"
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(train_directory, data_type)

    control = 'label'
    random_order = np.random.permutation(len(train))
    train['RandomOrder'] = random_order
    train = train.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    train = train.drop(columns=['RandomOrder'])

    # Validation
    val_directory = "src/vit_model/archive/validation/"
    val = pd.DataFrame()
    val['image'], val['label'] = load_dataset(val_directory, data_type)

    control = 'label'
    random_order = np.random.permutation(len(val))
    val['RandomOrder'] = random_order
    val = val.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    val = val.drop(columns=['RandomOrder'])

    # Test
    test_directory = "src/vit_model/archive/test/"
    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(test_directory, data_type)

    control = 'label'
    random_order = np.random.permutation(len(test))
    test['RandomOrder'] = random_order
    test = test.sort_values(by=['RandomOrder', control]).reset_index(drop=True)
    test = test.drop(columns=['RandomOrder'])

    train_features = extract_features(train['image'], image_size)
    test_features = extract_features(test['image'], image_size)
    val_features = extract_features(val['image'], image_size)

    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    y_val = le.transform(val['label'])

    # 2. Turn data into tensors using data loaders
    train_dataloader = DataLoader(list(zip(train_features, y_train)),
                                  batch_size=batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(list(zip(val_features, y_val)),
                                batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(list(zip(test_features, y_test)),
                                 batch_size=batch_size,
                                 shuffle=False)

    classes = ['Forest', 'Mountain']
    return train_dataloader, test_dataloader, val_dataloader, torch.from_numpy(y_train), torch.from_numpy(y_test),  torch.from_numpy(y_val), classes


def load_dataset(directory, data_type):
    """
    Load image dataset from a given directory.

    Args:
        directory (str): Directory path containing image files.
        data_type (str): Type of data files ('jpg' or 'tiff').

    Returns:
        tuple: Tuple containing image paths and corresponding labels.

    """
    if data_type == 'jpg':
        image_paths = []
        labels = []

        directory_ = os.listdir(directory)

        for i in directory_:
            # path = os.listdir(directory + folder)
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

    elif data_type == 'tiff':
        image_paths = []
        labels = []

        directory_ = os.listdir(directory)

        for i in directory_:
            # path = os.listdir(directory + folder)
            if i.startswith('.') or i.startswith('_'):
                directory_.remove(i)

        for folder in directory_:
            for filename in os.listdir(directory+folder):
                image_path = os.path.join(directory, folder, filename)

                if filename.startswith('.') or filename.startswith('_') or filename[-4:] != '.tif':
                    pass

                else:
                    image_paths.append(image_path)
                    labels.append(folder)

    return image_paths, labels


def extract_features(images, image_size):

    """
    Extract features from images.

    Args:
        images (list): List of image paths.
        image_size (int): Size of the input images.

    Returns:
        list: List of image features.

    """

    data_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()])

    features = []
    for image in tqdm(images):
        img = Image.open(image)
        img = img.convert('RGB')
        data = data_transform(img)
        features.append(data)
    return features

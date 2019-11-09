import pandas as pd
import numpy as np
import os

import cv2
import albumentations as albu
from albumentations import pytorch as AT

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

def get_dataset(train=True, path='../input/'):
    """
    Returns either the training or testing dataset in a DataFrame
    """
    if train:
        df = pd.read_csv(f'{path}train.csv')
    else:
        df = pd.read_csv(f'{path}sample_submission.csv')

    return df

def prepare_dataset(df):
    """
    Prepares dataset
    """
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df['im_id'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    return df

def get_train_test(df, test_size=0.2):
    """
    Copied from https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud
    """
    df = df[~df['EncodedPixels'].isnull()]
    df['Image'] = df['Image_Label'].map(lambda x: x.split('_')[0])
    df['Class'] = df['Image_Label'].map(lambda x: x.split('_')[1])

    classes = df['Class'].unique()
    df = df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        df[class_name] = df['Class'].map(lambda x: 1 if class_name in x else 0)
    class_id = df['Class'].map(lambda x: str(sorted(list(x))))

    train_imgs, val_imgs = train_test_split(
        df['Image'].values,
        test_size=test_size,
        stratify=class_id,
        random_state=1)
    
    return train_imgs, val_imgs

class CloudDataset(Dataset):
    """
    https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools
    """
    def __init__(self, df, datatype, img_ids, transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
    preprocessing=None, folder='../input/'):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{folder}train_images"
        else:
            self.data_folder = f"{folder}test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)


#### BELOW COPIED FROM  https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools ####

def get_img(x, folder: str='train_images', loc='../input/'):
    """
    Return image based on image name and folder.
    """
    data_folder = f"{loc}{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if isinstance(mask_rle, float):
        return np.zeros(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
            
    return masks


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())
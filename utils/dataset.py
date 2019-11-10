from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
import numpy as np
from PIL import Image
import os
from myargs import args
import torch
import json

DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)

eps = 1e-8


def findFile(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    # impath needs a backslash
    def __init__(self, impath, labelpath, eval):
        'Initialization'
        self.eval = eval

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

        if self.eval:
            type = 'val'
        else:
            type = 'train'

        with open('./bdd100k/labels/bdd100k_labels_images_{}.json'.format(type), 'r') as jsonfil:
            img_info = jsonfil.read()
        datalist = []
        img_info = json.loads(img_info)
        for image in img_info:
            datalist.append({
                'image': impath + image['name'],
                'label': labelpath + image['name'].replace('.jpg', '.png')
            })

        self.datalist = datalist

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = Image.open(self.datalist[index]['image'])
        label = Image.open(self.datalist[index]['label'])

        if not self.eval:
            image = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = func.crop(image, i, j, h, w)
        image = self.normalize(image)

        label = func.crop(label, i, j, h, w)
        label = np.asarray(label, dtype=np.uint8)
        label = torch.from_numpy(label).long()

        return image, label


def GenerateIterator(impath, labelpath, eval=False, shuffle=True):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': 0,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, labelpath, eval=eval), **params)


class Dataset_test(data.Dataset):
    'Characterizes a dataset for PyTorch'
    # impath needs a backslash
    def __init__(self, impath):
        'Initialization'
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

        path_list = findFile(impath, '.jpg')

        datalist = []
        for test_pth in path_list:
            datalist.append({
                'image': test_pth
            })

        self.datalist = datalist

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        img_path = self.datalist[index]['image']
        image = Image.open(img_path)

        image = transforms.RandomCrop(size=(224, 224))(image)
        image = self.normalize(image)

        return image, img_path.replace('test_resize', 'out')


def GenerateIterator_test(impath, shuffle=False):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': 0,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset_test(impath), **params)


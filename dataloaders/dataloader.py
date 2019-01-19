import torch
import numpy as np
import random
import pandas as pd
import pickle
import cv2
from PIL import Image
import torchvision.transforms as transforms

from dataloaders.common import get_image_pil
from torch.utils.data import DataLoader, Dataset

import imgaug as ia
from imgaug import augmenters as iaa
from dataloaders.augmenters import Identity, sometimes, often, ImgAugTransform, ResizeCV2, GaussianBlurCV2

from dataloaders.common import inin_w



class Model:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = self.find_definition()

    def find_definition(self):
        if self.dataset_name in dataloaders.datasets.__dict__:
            return dataloaders.datasets.__dict__[self.dataset_name]
        else:
            assert False, 'Cannot find dataset'
            

    def get_args(parser):
        parser.add('--num_workers', type=int, default=4,   help='Number of data loading workers.')
        parser.add('--batch_size',  type=int, default=64,  help='Batch size')
        parser.add('--num_samples_train',  type=int, default=10000, help='Image size')
        
        return self.dataset.get_args(loadparser)


    def get_dataloader(self, args, model_native_transform, part):
        
        if hasattr(self.dataset, 'get_dataloader'):
            return self.dataset.get_dataloader(args, model_native_transform, part)
        else:
            
            dataset = self.dataset.get_dataset(args, model_native_transform, part)

            sampler = torch.utils.data.RandomSampler(part_data, replacement=False)

            if args.num_samples_train != -1: 
                sampler = torch.utils.data.RandomSampler(part_data, replacement=True, num_samples=args.num_samples_train)

            return DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampler=sampler if part == 'train' else None,
                            pin_memory=True,
                            drop_last=False if part == 'train' else False,
                            shuffle=None if part == 'train' else False,
                            worker_init_fn=inin_w)





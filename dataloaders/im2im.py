import torch
import numpy as np
import pandas as pd
#from PIL import Image
import torchvision.transforms as transforms

from dataloaders.common import get_image_cv2, uint2float, inin_w
from dataloaders.augmenters import RandomCrop, ResizeCV2, Identity
from torch.utils.data import DataLoader, Dataset
# import imgaug as ia
from imgaug import augmenters as iaa

import os 
import json
import cv2


def get_args(parser):
    parser.add('--splits_dir',  type=str, default="",  help="path to directory with splits")
    
    parser.add('--num_workers', type=int, default=4,   help='number of data loading workers')
    parser.add('--batch_size',  type=int, default=16,  help='batch size')
    parser.add('--image_size',  type=int, default=256, help='image size')
    
    parser.add('--augment_train',     default=True,  action="store_bool")
    parser.add('--augment_test',      default=False, action="store_bool")

    parser.add('--resize600', default=False, action="store_bool")

    parser.add('--resize_train', default=True, action="store_bool")
    
    parser.add('--test_csv',    type=str, default="",  help="optionally override path to test.csv")

    parser.add('--params_stats_path',    type=str, default='',  help="")
    parser.add('--crop_or_resize',       type=str, default='resize')

    parser.add('--use_native_transform', default=True, action="store_bool")

    return parser

class ToTensor16(object):
    def __call__(self, pic):
        return torch.from_numpy(uint2float(pic.transpose(2, 0, 1)).astype(np.float32))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_dataloader(args, model_native_transform, part):
    part_data = get_part_data(args, part)

    augmenter = ImgAugTransform(args, part)

    shared_transform = transforms.Compose([
        augmenter,
    ])

    input_transform = transforms.Compose([
            ToTensor16(),
            model_native_transform if args.use_native_transform else Identity
    ])

    target_transform = transforms.Compose([
            ToTensor16(),
    ])

    dataset = PairsDataset(part_data, 
        shared_transform=shared_transform,
        input_transform =input_transform,
        target_transform=target_transform,
        args=args)

    return DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True if part == 'train' else False,
                    shuffle=True if part == 'train' else False,
                    worker_init_fn= inin_w)

# -------------------------
#         Functions
# -------------------------


class ImgAugTransform(object):
    def __init__(self, args, part):

        if part == 'train':
            if args.augment_train:

                # if args.no_resize_train:
                seq_geom = iaa.Sequential([
                    iaa.Fliplr(0.5),
                    # iaa.Flipud(0.5), 
                ])
                # else:


                seq_color = iaa.Sequential([
                ])
                
            else:
                seq_geom, seq_color = iaa.Noop(), iaa.Noop()

            if not args.resize_train:
                # print(1)
                cropper_or_resizer = ResizeCV2({"height":400, "width":600}, cv2.INTER_CUBIC)
            else:
                cropper_or_resizer = RandomCrop(args.image_size, shared_crop=True) if args.crop_or_resize == 'crop' else ResizeCV2({"height":args.image_size, "width":args.image_size}, cv2.INTER_CUBIC)
                

            # cropper_or_resizer = ResizeCV2({"height":args.image_size, "width":args.image_size}, cv2.INTER_CUBIC)
            if args.crop_or_resize == 'crop':
                cropper_or_resizer = RandomCrop(args.image_size) 

        else:
            seq_geom, seq_color = iaa.Noop(), iaa.Noop()

            if args.resize600 or (not args.resize_train):
                cropper_or_resizer = ResizeCV2({"height":400, "width":600}, cv2.INTER_CUBIC)
            else:
                cropper_or_resizer = RandomCrop(args.image_size, shared_crop=True) if args.crop_or_resize == 'crop' else ResizeCV2({"height":args.image_size, "width":args.image_size}, cv2.INTER_CUBIC)
                

        seq_geom = iaa.Sequential([
                    cropper_or_resizer,
                    seq_geom])

        self.seq_geom = seq_geom
        self.seq_color = seq_color
        
    def __call__(self, imgs):
        # Geom
        seq_geom_det  = self.seq_geom.to_deterministic()
        images_aug = [seq_geom_det.augment_image(x) for x in imgs]

        # !!!!
        # Do not use th following code:
        # !!!!
        # images_aug = seq_geom_det.augment_images(imgs)

        # Color
        seq_color_det = self.seq_color.to_deterministic()
        images_aug = [seq_color_det.augment_image(x) for x in images_aug]

        return images_aug


class PairsDataset(Dataset):
    def __init__(self, df, shared_transform=None, 
                           input_transform=None, 
                           target_transform=None, args=None):

        super(PairsDataset, self).__init__()
        self.df = df
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.shared_transform = shared_transform

        self.args = args
        
    def __getitem__(self, index):

        row = self.df.loc[index]

        # =====================
        #     Load images
        # =====================
        input  = np.array(get_image_cv2(row['img0_path']))
        target = np.array(get_image_cv2(row['img1_path']))
            
       
        if self.shared_transform is not None:
            (input, target) = self.shared_transform((input, target))
        
        if self.input_transform is not None:
            input = self.input_transform(input)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        

        return row['img0_path'], \
               target, \
               input
               

    def __len__(self):
        return self.df.shape[0]


def get_part_data(args, part):
    if part == 'test':
        if args.test_csv == "":
            test_csv_path = f"{args.splits_dir}/test.csv"
        else:
            test_csv_path  = args.test_csv

        return pd.read_csv(test_csv_path)

    else:
        return pd.read_csv(f"{args.splits_dir}/{part}.csv")

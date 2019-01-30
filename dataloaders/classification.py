import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from dataloaders.common import get_image, any2float, ToTensor, get_part_data
from dataloaders.augmenters import RandomCrop, ResizeCV2, Identity, ImgAugTransform, often, sometimes, GaussianBlurCV2
from torch.utils.data import DataLoader, Dataset
from dataloaders.augmenters import Rot90

from imgaug import augmenters as iaa
import imgaug as ia
import os 
import json
import cv2
from functools import lru_cache



class Dataset(object):

        
    @staticmethod
    def get_args(parser):
        
        
        parser.add('--splits_dir',      default='', type=str)

        parser.add('--image_size',  type=int, default=256, help='image size')
    
        parser.add('--train_phase_mode',     default='train=crop,augment^val=crop',  action="store_bool")
        parser.add('--test_phase_mode',      default='crop',  action="store_bool")

        parser.add('--crop_or_resize',       type=str, default='resize')

        parser.add('--use_native_transform', default=True, action="store_bool")
        
        parser.add('--target_columns',  type=str, default="")

        return parser

    @staticmethod
    def get_dataset(args, model_native_transform, part, phase):


        part_data = get_part_data(args, part)
        

        target_columns = args.target_columns.split(',')


        # Get how many classes are there
        if 'num_classes' not in vars(args) or args.num_classes == "":
            train_data = get_part_data(args, 'train')
            merged_data = pd.concat([train_data, part_data], axis=0, ignore_index=True)
            args.num_classes = ','.join([str(merged_data.loc[merged_data[x] >= 0, x].max() + 1) for x in target_columns])


        augmenter = get_transform(args, phase)

        input_transform = transforms.Compose([
                augmenter,
                ToTensor(),
                model_native_transform if args.use_native_transform else Identity
        ])


        dataset = CsvDataset(part_data, 
            input_transform =input_transform,
            target_columns=target_columns,
            args=args)


        return dataset






class CsvDataset(Dataset):
    def __init__(self, df, input_transform=None, target_columns=None, args=None, treat_as_all_multiclass=True):

        super().__init__()
        self.df = df
        
        self.input_transform = input_transform
        self.target_columns = target_columns

        self.treat_as_all_multiclass= treat_as_all_multiclass
        self.args = args

    
    def __getitem__(self, index):

        row = self.df.loc[index]


        input  = get_image(row['img_path'])


        target = list(row[self.target_columns].values)
        if not self.treat_as_all_multiclass:
            target = [np.array(target)]

        if self.input_transform:
            input = self.input_transform(input)

        return { 
                    'names': row['img_path'],
                    'target': target,
                    'input': input
                }
               

    def __len__(self):
        return self.df.shape[0]





# -------------------------
#         Transforms
# -------------------------


def get_transform(args, phase):

    d = {
        'train': train_transform,
        'val':   val_transform,
        'test':  test_transform,    
    }

    return d[phase](args)



def train_transform(args):


    seq_geom = iaa.Sequential([
                    iaa.Fliplr(0.5),
                    
                    sometimes(iaa.Add((-10, 10))), # change brightness of images (by -10 to 10 of original value)
                    sometimes(iaa.AddToHueAndSaturation((-10, 8))), # change hue and saturation
                # #         (iaa.PiecewiseAffine(scale=(0.00, 0.02), order=3)),
                #     sometimes(iaa.ContrastNormalization((0.5, 1.5))),
                    sometimes(GaussianBlurCV2(sigma=(0, 0.5))),  # blur images with a sigma of 0 to 3.0

                    # often(iaa.CropAndPad(
                    #         percent=(-0.05, 0.05),
                    #         pad_mode=ia.ALL,
                    #         pad_cval=(0, 255)
                    #     )),
                    often(iaa.Affine(
                            scale={"x": (0.85, 1.0), "y": (0.85, 1.0)}, # scale images to 80-120% of their size, individually per axis
                            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}, # translate by -20 to +20 percent (per axis)
                            rotate=(-35, 35),  # rotate by -45 to +45 degrees
                            shear=(-14, 14),  # shear by -16 to +16 degrees
                            order=[0, 1, 3], # use nearest neighbour or bilinear interpolation (fast)
                            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                            mode=ia.ALL, # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                            backend='cv2'
                        )),
           
                #         (iaa.ElasticTransformation(alpha=(0.0, 1.5), sigma=(9.0, 10))),
                ])
    
    if args.crop_or_resize == 'crop':
        cropper_or_resizer = RandomCrop(args.image_size, shared_crop=True)
    else:
        cropper_or_resizer = ResizeCV2({"height":args.image_size, "width":args.image_size}, cv2.INTER_AREA)

    seq_geom = iaa.Sequential([
                    cropper_or_resizer,
                    seq_geom
                    ])


    return ImgAugTransform(geom_transform=seq_geom)





def val_transform(args):

    if args.crop_or_resize == 'crop':
        cropper_or_resizer = RandomCrop(args.image_size, shared_crop=True)
    else:
        cropper_or_resizer = ResizeCV2({"height":args.image_size, "width":args.image_size}, cv2.INTER_AREA)



    return ImgAugTransform(geom_transform=cropper_or_resizer)

   



def test_transform(args):
    return val_transform(args)



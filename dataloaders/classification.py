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

def get_args(parser):
    parser.add('--splits_dir',  type=str, default="",  help="Path to a directory with splits csv.")
    parser.add('--num_workers', type=int, default=4,   help='Number of data loading workers.')
    parser.add('--batch_size',  type=int, default=64,  help='Batch size')
    parser.add('--image_size',  type=int, default=256, help='Image size')
    parser.add('--augment_train', default=True, action="store_bool", help='Whether to augment images during training.')
    parser.add('--augment_test',  default=False, action="store_bool", help='Whether to augment images during testing.')

    parser.add('--target_columns',  type=str, default="",  help="")
    # parser.add('--crop_or_resize', type=str, help='Whether to augment images during training.')
    
    parser.add('--use_native_transform', default=True, action="store_bool")
    parser.add('--augmenter', type=str, default="img_aug")

    return parser

def get_dataloader(args, model_native_transform, part):
    part_data = get_part_data(args, part)
    target_columns = args.target_columns.split(',')

    # Get how many classes are there
    train_data = get_part_data(args, 'train')
    merged_data = pd.concat([train_data, part_data], axis=0, ignore_index=True)
    if args.num_classes == "":
        args.num_classes = ','.join([str(merged_data.loc[merged_data[x] >= 0, x].max() + 1) for x in target_columns])
    
    # Define augmenter
    augmenter = ImgAugTransform(get_seq(args, part))
    

    need_augment = (args.augment_test and part != 'train') or (args.augment_train and part == 'train')
    input_transform = transforms.Compose([
            ImgAugTransform(ResizeCV2({"height":args.image_size, "width":args.image_size}, cv2.INTER_CUBIC)),
            augmenter if need_augment else Identity,
            transforms.ToTensor(),
            model_native_transform if args.use_native_transform else Identity
    ])

    
    dataset = CsvDataset(part_data, target_columns, input_transform=input_transform)
   
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

def get_seq(args, part):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        # often(iaa.CropAndPad(
        #         percent=(-0.05, 0.05),
        #         pad_mode=ia.ALL,
        #         pad_cval=(0, 255)
        #     )),
        often(iaa.Affine(
                scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
                rotate=(-35, 35),  # rotate by -45 to +45 degrees
                shear=(-14, 14),  # shear by -16 to +16 degrees
                order=[0, 1, 3], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL, # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                backend='cv2'
            )),
    #     sometimes(iaa.Add((-10, 10))), # change brightness of images (by -10 to 10 of original value)
    #     sometimes(iaa.AddToHueAndSaturation((-10, 8))), # change hue and saturation
    # #         (iaa.PiecewiseAffine(scale=(0.00, 0.02), order=3)),
    #     sometimes(iaa.ContrastNormalization((0.5, 1.5))),
        sometimes(GaussianBlurCV2(sigma=(0, 0.5))),  # blur images with a sigma of 0 to 3.0
    #         (iaa.ElasticTransformation(alpha=(0.0, 1.5), sigma=(9.0, 10))),
    ])

    return seq

class CsvDataset(Dataset):
    def __init__(self, df, target_columns, input_transform=None, treat_as_all_multiclass=True):
        super(CsvDataset, self).__init__()
        self.df = df
        self.target_columns = target_columns
        
        self.treat_as_all_multiclass = treat_as_all_multiclass
        self.input_transform = input_transform
        # self.target_transform = target_transform

    def __getitem__(self, index):

        row = self.df.loc[index]

        input  = get_image_pil(row['img_path'])

        target = list(row[self.target_columns].values)
        if not self.treat_as_all_multiclass:
            target = [np.array(target)]

        if self.input_transform:
            input = self.input_transform(input)

        # if self.target_transform:
        #     target = self.target_transform(target)

        return (row['img_path'], input, *target)

    def __len__(self):
        return self.df.shape[0]

# from torch.utils.data.dataloader import default_collate
# def my_collate(batch):

#     d_idxs = [x[-1] for x in batch]
#     d_idx_max = max(d_idxs) 


#     per_d = [np.where(np.array(d_idxs) == i)[0] for i in range(d_idx_max + 1)]
#     # print(per_d)
#     out = []
#     for i in range(len(batch[0])):
        
#         out.append ( [ default_collate([batch[j][i] for j in idxs]) if len(idxs) > 0  else None for idxs in per_d ] )
    
#     # [if bat]
#     # print(len(batch), len(batch[0]))

#     # ff = [([x[i] for x in batch if x[i] is not None]) for i in range(len(batch[0]))]

#     # print (batch[0])
#     # print (len(ff[0]))
#     # for i in range(len(batch[0])):
#     #     print(len([x[i] for x in batch if x[i] is not None]))
#     return out[0], out[1], out[2]


def get_part_data(args, part):
    return pd.read_csv(f"{args.splits_dir}/{part}.csv")

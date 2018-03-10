import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from .common import get_image_pil
from torch.utils.data import DataLoader, Dataset

import imgaug as ia
from imgaug import augmenters as iaa

def get_args(parser):
    parser.add('--splits_dir',  type=str, default="",  help="path to directory with splits")
    parser.add('--num_workers', type=int, default=4,   help='number of data loading workers')
    parser.add('--batch_size',  type=int, default=64,  help='batch size')
    parser.add('--image_size',  type=int, default=224, help='image size')
    parser.add('--augment',     default=False, action="store_true")

    return parser

def get_dataloader(args, part):
    train_df, val_df, test_df, target_columns, preprocessor = get_dfs(args)

    if part == 'train':
        Identity = transforms.Lambda(lambda x: x)
        train_transform = transforms.Compose([
            ImgAug(seq) if args.augment else Identity,
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
        ])

        dataloader_train = setup_dataset(train_df, target_columns, train_transform, args.batch_size, args.num_workers)
        return dataloader_train

    elif part == 'val':
        val_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
        ])

        dataloader_val = setup_dataset(val_df, target_columns, val_transform, args.batch_size, args.num_workers, drop_last=False, shuffle=False)
        return dataloader_val

    elif part == 'test':
        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
        ])

        dataloader_test = setup_dataset(test_df, target_columns, test_transform, args.batch_size, args.num_workers, drop_last=False, shuffle=False)
        return dataloader_test





# -------------------------
#         Functions
# -------------------------


sometimes = lambda aug: iaa.Sometimes(0.4, aug)
often = lambda aug: iaa.Sometimes(0.8, aug)

seq = iaa.Sequential([
    
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    often(iaa.Affine(
        # scale images to 80-120% of their size, individually per axis
        scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
        # translate by -20 to +20 percent (per axis)
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-8, 8),  # rotate by -45 to +45 degrees
        shear=(-5, 5),  # shear by -16 to +16 degrees
        order=[3],  # use nearest neighbour or bilinear interpolation (fast)
        cval=(0),  # if mode is constant, use a cval between 0 and 255
        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        mode='constant'
    )),
    sometimes(iaa.Add((-8, 8))), # change brightness of images (by -10 to 10 of original value)
    sometimes(iaa.AddToHueAndSaturation((-8, 8))), # change hue and saturation
    sometimes(iaa.PiecewiseAffine(scale=(0.00, 0.02), order=3)),
    sometimes(iaa.ContrastNormalization((0.5, 1.5))),
    often(iaa.GaussianBlur(sigma=(0, 0.3))),  # blur images with a sigma of 0 to 3.0
#     (iaa.ElasticTransformation(alpha=(0.0, 1.5), sigma=(9.0, 10))),
    
])



class ImgAug(object):
    def __init__(self, augmenter_pipeline=None):
        self.augmenter_pipeline = augmenter_pipeline
        
    def __call__(self, img):
        
        if self.augmenter_pipeline is not None:
            img = Image.fromarray(self.augmenter_pipeline.augment_image(np.array(img)))

        return img



class FolderWithImages(Dataset):
    def __init__(self, df, target_columns, input_transform=None, target_transform=None):
        super(FolderWithImages, self).__init__()
        self.df = df
        self.target_columns = target_columns
        
        self.target_columns_mask = [x + '_mask' for x in target_columns]
        if not all([x in df.columns for x in self.target_columns_mask]):
            print("Masks not found.")
            self.target_columns_mask = self.target_columns

        self.input_transform = input_transform
        self.target_transform = target_transform

        # self.masked = False

    def __getitem__(self, index):

        row = self.df.loc[index]

        input  = get_image_pil(row['img_path'])
        target = np.array(list(row[self.target_columns].values))
        mask   = np.array(list(row[self.target_columns_mask].values))

        if self.input_transform:
            input = self.input_transform(input)


        if self.target_transform:
            target = self.target_transform(target)

        # print(target, mask)
        return row['img_path'], input, target

    def __len__(self):
        return self.df.shape[0]


def get_dfs(args):

    # Read splits info
    train_df = pd.read_csv(f"{args.splits_dir}/train.csv")
    val_df   = pd.read_csv(f"{args.splits_dir}/val.csv")
    test_df  = pd.read_csv(f"{args.splits_dir}/test.csv")

    target_columns = ['label']
    args.n_classes = [train_df.label.nunique()]
        
    preprocessor = None

    return train_df, val_df, test_df, target_columns, preprocessor


def setup_dataset(df, target_columns, input_transform, batch_size, num_workers, shuffle=True, drop_last=True):
    dataset = FolderWithImages(
        df, target_columns, input_transform=input_transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=int(num_workers),
                            pin_memory=True,
                            drop_last=drop_last)
    return dataloader





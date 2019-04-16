import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from dataloaders.common import get_image_pil
from dataloaders.augmenters import RandomCrop
from torch.utils.data import DataLoader, Dataset

import imgaug as ia
from imgaug import augmenters as iaa

import os 




def get_args(parser):
    parser.add('--splits_dir',  type=str, default="",  help="path to directory with splits")
    
    parser.add('--num_workers', type=int, default=5,   help='number of data loading workers')
    parser.add('--batch_size',  type=int, default=64,  help='batch size')
    parser.add('--image_size',  type=int, default=224, help='image size')
    
    parser.add('--augment',     default=False, action="store_true")
    parser.add('--augment_test',default=False, action="store_true")

    parser.add('--target_columns', type=str, default='label')

    parser.add('--test_csv',    type=str, default="",  help="optionally override path to test.csv")

    return parser

def get_dataloader(args, part):
    train_df, val_df, test_df = get_dfs(args)

    args.use_cond = False

    if part == 'train':   
        
        train_transform = ImgAug(args)
        dataloader_train = setup_dataset(train_df, train_transform, args.batch_size, args.num_workers, use_cond = args.use_cond)
        return dataloader_train

    elif part == 'val':
        
        val_transform = ImgAugTest(args)
        dataloader_val = setup_dataset(val_df, val_transform, args.batch_size, args.num_workers, drop_last=False, shuffle=False, use_cond = args.use_cond)
        return dataloader_val

    elif part == 'test':

        test_transform = ImgAugTest(args)
        dataloader_test = setup_dataset(test_df, test_transform, args.batch_size, args.num_workers, drop_last=False, shuffle=False, use_cond = args.use_cond)
        return dataloader_test



# -------------------------
#         Functions
# -------------------------


sometimes = lambda aug: iaa.Sometimes(0.4, aug)
often = lambda aug: iaa.Sometimes(0.8, aug)



class ImgAug(object):
    def __init__(self, args):
        if args.augment:

            seq_geom = iaa.Sequential([
                RandomCrop(args.image_size, shared_crop=True),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5), 
            ])

            seq_color = iaa.Sequential([                
            ])
            
        else:
            seq_geom = iaa.Sequential([
                iaa.Scale({"height":args.image_size, "width":args.image_size}, 'cubic')
            ]) 

            seq_color = iaa.Sequential([])


        self.seq_geom = seq_geom
        self.seq_color = seq_color
        
    def __call__(self, imgs):
        seq_geom_det  = self.seq_geom.to_deterministic()
        seq_color_det = self.seq_color.to_deterministic()

        images_aug = seq_geom_det.augment_images(imgs)

        images_aug[0] = seq_color_det.augment_images([images_aug[0]])[0]

        return images_aug


class ImgAugTest(object):
    def __init__(self, args):
        seq = iaa.Sequential([
            RandomCrop(args.image_size, shared_crop=True)
        ])

        self.seq = seq
        
    def __call__(self, imgs):
        
        seq_det = self.seq.to_deterministic()
        images_aug = seq_det.augment_images(imgs)

        return images_aug




class PairsDataset(Dataset):
    def __init__(self, df, input_transform=None, target_transform=None, use_cond=False):
        super(PairsDataset, self).__init__()
        self.df = df
        
        self.input_transform = input_transform
        self.target_transform = target_transform

        # self.sigma = sigma
        self.use_cond = use_cond

    def __getitem__(self, index):

        row = self.df.loc[index]


        # pose_path = f'{dirname}/1/{os.path.basename(row["img_path"])[:-4]}_rendered.png'
        # seg_path  = f'{dirname}/segs/{os.path.basename(row["img_path"])[:-4]}.png'
        

        # if self.use_cond:
        #     # imgs = [
        #     #     np.array(get_image_pil(row['img_path'])),
        #     #     np.array(get_image_pil(pose_path)),
        #     #     np.array(get_image_pil(seg_path)),
        #     # ]
        #     assert False
        # else:
        imgs = [
            np.array(get_image_pil(row['img0_path'])),
            np.array(get_image_pil(row['img1_path']))
        ]
        # kps = _get_keypoints(row[self.target_columns])

        if self.input_transform is not None:
            imgs_aug = self.input_transform(imgs)
        else:
            imgs_aug = imgs


        cond = np.array(n_params,)
        input = np.concatenate([ (imgs_aug[0].transpose(2, 0, 1)/255.).astype(np.float32),
                                 cond], 0)

        # print(input.shape)
        # heatmaps, flags = _generate_heatmap(img_aug, kps_aug, sigma=self.sigma, background=False)

        # target = np.array(list(row[self.target_columns].values))

        # print(np.max(input), np.min(input))
        return row['img0_path'],\
               (imgs_aug[0].transpose(2, 0, 1)/255.).astype(np.float32), \
               (imgs_aug[1].transpose(2, 0, 1)/255.).astype(np.float32)

    def __len__(self):
        return self.df.shape[0]





def get_dfs(args):

    # Read splits info
    train_df = pd.read_csv(f"{args.splits_dir}/train.csv")
    val_df   = pd.read_csv(f"{args.splits_dir}/val.csv")

    try:
        if args.test_csv == "":
            test_df  = pd.read_csv(f"{args.splits_dir}/test.csv")
        else:
            test_df  = pd.read_csv(args.test_csv)
            
    except Exception as e:
        test_df = None


    return train_df, val_df, test_df


def setup_dataset(df, input_transform, batch_size, num_workers, shuffle=True, drop_last=True, use_cond=False):
    dataset = PairsDataset(
        df, input_transform=input_transform, use_cond = use_cond)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=int(num_workers),
                            pin_memory=True,
                            drop_last=drop_last)
    return dataloader
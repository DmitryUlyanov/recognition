import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from dataloaders.common import get_image_pil
from torch.utils.data import DataLoader, Dataset

import imgaug as ia
from imgaug import augmenters as iaa

def get_args(parser):
    parser.add('--splits_dir',  type=str, default="",  help="path to directory with splits")
    parser.add('--num_workers', type=int, default=4,   help='number of data loading workers')
    parser.add('--batch_size',  type=int, default=64,  help='batch size')
    parser.add('--image_size',  type=int, default=512, help='image size')

    parser.add('--augment',     default=False, action="store_true")
    parser.add('--augment_test',default=False, action="store_true")

    parser.add('--normalize', default=False, action="store_true")
    parser.add('--augmenter', type=str, default="img_aug")

    parser.add('--sigma', type=float, default=9)

    parser.add('--target_columns', type=str, default='')
    return parser

class ImgAug(object):
    def __init__(self, args):
        seq = iaa.Sequential([
    
            # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            often(iaa.CropAndPad(
                    percent=(-0.05, 0.05),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
            often(iaa.Affine(
                    scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-25, 25), # rotate by -45 to +45 degrees
                    shear=(-12, 12), # shear by -16 to +16 degrees
                    order=[0, 1, 3], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL, # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2'
                )),
            sometimes(iaa.Add((-10, 10))), # change brightness of images (by -10 to 10 of original value)
            sometimes(iaa.AddToHueAndSaturation((-10, 8))), # change hue and saturation
        #         (iaa.PiecewiseAffine(scale=(0.00, 0.02), order=3)),
            sometimes(iaa.ContrastNormalization((0.5, 1.5))),
            sometimes(iaa.GaussianBlur(sigma=(0.0, 2.0))),  # blur images with a sigma of 0 to 3.0
        #         (iaa.ElasticTransformation(alpha=(0.0, 1.5), sigma=(9.0, 10))),
            iaa.Scale({"height":args.image_size, "width":args.image_size}, 'cubic')
        ])


        self.seq = seq
        
    def __call__(self, img, keypoints):
        
        return augment(img, keypoints, self.seq)


class ImgAugTest(object):
    def __init__(self, args):
        seq = iaa.Sequential([
            iaa.Scale({"height":args.image_size, "width":args.image_size}, 'cubic')
        ])


        self.seq = seq
        
    def __call__(self, img, keypoints):
        
        return augment(img, keypoints, self.seq)


def get_dataloader(args, part):
    train_df, val_df, test_df = get_dfs(args)
    target_columns = args.target_columns.split(',')
    args.num_output_channels = len(target_columns)
    
    Identity = transforms.Lambda(lambda x: x)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # if args.augmenter == 'pytorch': 
    #     augmenter = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #     ])
    # elif args.augmenter == 'img_aug':
    #     augmenter = ImgAug(seq)
    # else:
    #     assert False

    if part == 'train':   
        # train_transform = transforms.Compose([
        #     augmenter if args.augment else Identity,
        #     # transforms.Resize([args.image_size, args.image_size]),
        #     transforms.ToTensor(),
        #     normalize if args.normalize else Identity
        # ])

        train_transform = ImgAug(args)
        dataloader_train = setup_dataset(train_df, target_columns, train_transform, args.batch_size, args.num_workers, args.sigma)
        return dataloader_train

    elif part == 'val':
        # val_transform = transforms.Compose([
        #     augmenter if args.augment_test else Identity,
        #     # transforms.Resize([args.image_size, args.image_size]),
        #     transforms.ToTensor(),
        #     normalize if args.normalize else Identity
        # ])

        val_transform = ImgAugTest(args)
        dataloader_val = setup_dataset(val_df, target_columns, val_transform, args.batch_size, args.num_workers, args.sigma, drop_last=False, shuffle=False)
        return dataloader_val

    elif part == 'test':
        # test_transform = transforms.Compose([
        #     augmenter if args.augment_test else Identity,
        #     # transforms.Resize([args.image_size, args.image_size]),
        #     transforms.ToTensor(),
        #     normalize if args.normalize else Identity
        # ])

        test_transform = ImgAugTest(args)
        dataloader_test = setup_dataset(test_df, target_columns, test_transform, args.batch_size, args.num_workers, args.sigma, drop_last=False, shuffle=False)
        return dataloader_test





# -------------------------
#         Functions
# -------------------------


sometimes = lambda aug: iaa.Sometimes(0.4, aug)
often = lambda aug: iaa.Sometimes(0.8, aug)



def _get_keypoints(keypoints_list_str):
    
    kpts = []
    for i, kpt_str in enumerate(keypoints_list_str):
        
        [x_str, y_str, vis_str] = kpt_str.split('_')
        
        if vis_str != '-1':
            kpts.append([float(x_str), float(y_str)])
        else:
            kpts.append([-1, -1])
            
    return np.array(kpts)

def _generate_heatmap(img, keypoints, sigma, background=False):
    '''
        keypoints is an iterable of tuples (x, y) with -1 in x, y if the keypoint should be excluded
    '''
    height, width, _ = img.shape
    
    heatmaps = np.zeros((len(keypoints) + int(background), height, width), dtype=np.float32) # (24 points + background)
        
    
    X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    meshgrid = np.concatenate([X[None,:], Y[None,:]])
    
    flags = np.ones(heatmaps.shape[0])
    for i, kpt in enumerate(keypoints):
        if np.all(kpt == -1):  # not labeled
            flags[i] = 0
            continue
        
        d = meshgrid - np.array(kpt)[:, None, None]
        d = (d**2).sum(0) / (2.0 * sigma * sigma)
        
        heatmaps[i, :, :][d < 4.6052] = np.exp(-d[d < 4.6052])
            
    if background:
        heatmaps[-1, :, :] = 1.0 - np.max(heatmaps[:-1, :, :], axis=0)  # for background
        
    return np.minimum(heatmaps, 1.0), flags


def augment(img, keypoints_array, seq):
    
    # Format keypoints in img_aug format
    kps_imgaug =  [ia.Keypoint(x=x[0], y=x[1]) for x in keypoints_array if x[0] >=0]
    keypoints_on_images = ia.KeypointsOnImage(kps_imgaug, shape=img.shape)
    
    seq_det = seq.to_deterministic()
    
    # Augment
    images_aug = seq_det.augment_images([img])
    keypoints_aug = seq_det.augment_keypoints([keypoints_on_images])
    
    # Convert keypoints back to np.array
    augmented_keypoints = np.array([[k.x, k.y] for k in keypoints_aug[0].keypoints])
    
    kps = keypoints_array.copy()
    kps[kps[:, 0] >= 0, :] = augmented_keypoints
    
    return images_aug[0], kps

class KeypointDataset(Dataset):
    def __init__(self, df, target_columns, sigma, input_transform=None, target_transform=None):
        super(KeypointDataset, self).__init__()
        self.df = df
        self.target_columns = target_columns
        
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.sigma = sigma
        # self.masked = False

    def __getitem__(self, index):

        row = self.df.loc[index]

        input  = np.array(get_image_pil(row['img_path']))
        kps = _get_keypoints(row[self.target_columns])

        if self.input_transform is not None:
            img_aug, kps_aug = self.input_transform(input, kps[:,:2])
        else:
            img_aug, kps_aug = input, kps[:, :2]

        heatmaps, flags = _generate_heatmap(img_aug, kps_aug, sigma=self.sigma, background=False)



        return row['img_path'], (img_aug.transpose(2, 0, 1)/255.).astype(np.float32), heatmaps, flags

    def __len__(self):
        return self.df.shape[0]


def get_dfs(args):

    # Read splits info
    train_df = pd.read_csv(f"{args.splits_dir}/train.csv")
    val_df   = pd.read_csv(f"{args.splits_dir}/val.csv")

    try:
        test_df  = pd.read_csv(f"{args.splits_dir}/test.csv")
    except Exception as e:
        test_df = None


    return train_df, val_df, test_df


def setup_dataset(df, target_columns, input_transform, batch_size, num_workers, sigma, shuffle=True, drop_last=True):
    dataset = KeypointDataset(
        df, target_columns, input_transform=input_transform, sigma=sigma)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=int(num_workers),
                            pin_memory=True,
                            drop_last=drop_last)
    return dataloader




import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from .common import get_image_pil
from torch.utils.data import DataLoader, Dataset

def get_args(parser):
    parser.add('--splits_dir',  type=str, default="",  help="path to directory with splits")
    parser.add('--num_workers', type=int, default=4,   help='number of data loading workers')
    parser.add('--batch_size',  type=int, default=64,  help='batch size')
    parser.add('--image_size',  type=int, default=224, help='image size')

    return parser

def get_dataloaders(args):
    train_df, val_df, test_df, target_columns, preprocessor = get_dfs(args)

    train_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
    ])

    dataloader_train = setup_dataset(train_df, target_columns, train_transform, args.batch_size, args.num_workers)
    dataloader_val = setup_dataset(val_df, target_columns, val_transform, args.batch_size, args.num_workers, drop_last=False, shuffle=False)

    return dataloader_train, dataloader_val



# -------------------------
#         Functions
# -------------------------


# def sometimes(aug):
#     return iaa.Sometimes(0.99, aug)


# seq = iaa.Sequential([
#     # sometimes(iaa.Crop(px=(0, 50))), # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.Fliplr(0.5),  # horizontally flip 50% of the images
#     sometimes(iaa.Affine(
#         # scale images to 80-120% of their size, individually per axis
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         # translate by -20 to +20 percent (per axis)
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-45, 45),  # rotate by -45 to +45 degrees
#         shear=(-16, 16),  # shear by -16 to +16 degrees
#         order=[0],  # use nearest neighbour or bilinear interpolation (fast)
#         cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
#         # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         mode='reflect'
#     )),
#     iaa.GaussianBlur(sigma=(0, 0.3))  # blur images with a sigma of 0 to 3.0
# ])


class ImagePreprocessor(object):
    def __init__(self, read_mode=-1, augmenter_pipeline=None,
                 crop_size=(400, 400), target_size=(224, 224)):
        self.read_mode = read_mode
        self.augmenter_pipeline = augmenter_pipeline
        self.crop_size = crop_size
        self.target_size = target_size
        self.normalization_factor = 2 ** 8

    @staticmethod
    def random_crop(img, crop_size):
        x = np.random.randint(img.shape[0] - crop_size[0] + 1)
        y = np.random.randint(img.shape[1] - crop_size[1] + 1)
        return img[x:x + crop_size[0], y:y + crop_size[1]]

    @staticmethod
    def fix_alpha(img):
        return img[:, :, :3]

    def __call__(self, img):
        img = self.random_crop(img, self.crop_size)
        img = cv2.resize(img, self.target_size)

        if self.augmenter_pipeline is not None:
            img = self.augmenter_pipeline.augment_image(img)

        img = img / self.normalization_factor
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

        input = get_image_pil(row['img_path'])
        target = np.array(list(row[self.target_columns].values))
        mask = np.array(list(row[self.target_columns_mask].values))

        if self.input_transform:
            input = self.input_transform(input)


        if self.target_transform:
            target = self.target_transform(target)

        # print(target, mask)
        return input, target, mask

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





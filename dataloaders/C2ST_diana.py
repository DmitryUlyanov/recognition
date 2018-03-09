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

def process_lfw_image(img):
    return Image.fromarray(np.array(img)[35:35+160,10:10+160,:])

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

    def __getitem__(self, index):

        row = self.df.loc[index]

        input = get_image_pil(row['img_path'])
        target = np.array(list(row[self.target_columns].values))
        mask = np.array(list(row[self.target_columns_mask].values))

        if self.input_transform:
            if target[0] == 1:
                input = process_lfw_image(input)
                # print(input.size)
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
    val_df   = pd.read_csv(f"{args.splits_dir}/test.csv")
    test_df  = pd.read_csv(f"{args.splits_dir}/test.csv")

    target_columns = ['label']
    args.n_classes = [2]
        
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




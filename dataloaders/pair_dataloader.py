import cv2
import glob
from os import path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os 


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1))


def to_numpy(img):
    return img.numpy().transpose(1, 2, 0)


def get_args(parser):
    parser.add('--pairs_dir', type=str, default="", help="path to directory with ground truth and raw images")
    parser.add('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add('--batch_size', type=int, default=64, help='batch size')
    parser.add('--image_size', type=int, default=224, help='image size')
    parser.add('--augment', default=False, action="store_true")
    parser.add('--gt_suffix', default='.jpg')
    parser.add('--raw_suffix', default='.jpg')

    parser.add('--dir_before', type=str, default="", help="path to directory with ground truth and raw images")
    parser.add('--dir_after', type=str, default="", help="path to directory with ground truth and raw images")

    parser.add('--as-is', default=False, help='left image channels number and channel depth unchanged')

    return parser


def load_image(path: str, as_is: bool):
    flags = []
    if as_is:
        flags.append(cv2.IMREAD_UNCHANGED)
    else:
        flags.append(cv2.IMREAD_COLOR)

    img = cv2.imread(path, *flags)
    img = cv2.resize(img, (600, 400))
    # OpenCV stores channels in BGR order. So we flip em
    img = img[:, :, ::-1]

    max_value = np.iinfo(img.dtype).max

    return (img / max_value).astype(np.float32)


def save_image(img, path):
    if type(img) == torch.Tensor:
        img = 255 * img.detach()
        img = to_numpy(img).astype(np.uint8)

    # OpenCV stores channels in BGR order. So we flip em
    img = img[:, :, ::-1]

    cv2.imwrite(path, img)


class PairsDir(Dataset):
    def __init__(self, dir_before: str, dir_after: str, raw_suffix: str, gt_suffix: str, as_is: bool):
        raw_files = sorted(glob.glob(path.join(dir_before, f'*{raw_suffix}')))  # type: List[str]
        gt_files = sorted(glob.glob(path.join(dir_after, f'*{gt_suffix}')))  # type: List[str]

       
        self.pairs = list(zip(raw_files, gt_files))
        # print(raw_files[0], gt_files[0])

        # raw_prefixes = set(map(lambda path_to_file: os.path.basename(path_to_file), raw_files))
        # gt_prefixes = set(map(lambda path_to_file: os.path.basename(path_to_file), gt_files))

        # self.names = list(raw_prefixes.intersection(gt_prefixes))

        # self.pairs = [[f'{dir_before}/{x}', f'{dir_after}/{x}' ] for x in self.names]
        # print( len(pairs))
        self.as_is = as_is
        self.raw_suffix = raw_suffix
        self.gt_suffix = gt_suffix

        self.transform = transforms.Resize(224)

    def __getitem__(self, index):
        pair = self.pairs[index]

        inp = transform(load_image(pair[0], self.as_is)
        target = load_image(pair[1], self.as_is)
        # print(pair)
        return pair[0], to_tensor(inp), to_tensor(target)

    def __len__(self):
        return len(self.pairs)


def get_dataloader_human(pairs_dir, raw_suffix, gt_suffix, batch_size=1,
                         num_workers=1, as_is=False):
    return DataLoader(PairsDir(pairs_dir, raw_suffix, gt_suffix, as_is),
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True)


def get_dataloader(args, part):
    return DataLoader(PairsDir(args.dir_before, args.dir_after, '', '', True),
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True)

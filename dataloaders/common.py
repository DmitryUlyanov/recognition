from huepy import red 
from PIL import Image
from torch.utils.data.dataloader import default_collate
import collections
import cv2
import numpy as np 
import random 
import torch
import pandas as pd 

try:
    import jpeg4py                    
except ImportError:
    has_jpeg4py = False
    print('jpegturbo is unavailable')
else:
    has_jpeg4py =True


def get_image(image_name):
    # if datapath is not None:
    #     image_name = (
    #         image_name if image_name.startswith(datapath) else
    #         os.path.join(datapath, image_name)
    #     )

    img = None

    if has_jpeg4py and image_name.endswith(("jpg", "JPG", "jpeg", "JPEG")):
        try:
            img = jpeg4py.JPEG(image_name).decode()
        except Exception:
            print('jpeg4py error!')
            pass


    if img is None:
        img = cv2.imread(image_name)

        if img is None:
            print(red(path))

        if len(img.shape) == 3:  # BGR -> RGB
            img = img[:, :, ::-1]

    if len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    if img.shape[-1] != 3 and not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in [".png", ".jpg", ".jpeg"])



def get_image_cv2(path, force_3channels=False):
    img = cv2.imread(path, -1)
    if img is None:
        print(red(path))
    if len(img.shape) == 3:
        img = img[:, :, :3]
        img = img[:, :, ::-1]
    elif force_3channels:
        img = np.concatenate([img[:, :, None], img[:, :, None], img[:, :, None]], axis=2)
        
    return img



def get_image_jpegturbo(path):
    return jpeg4py.JPEG(path)


def get_image_pil(path):
    img = Image.open(path).convert('RGB')
    return img

def inin_w(worker_id):
    np.random.seed(random.randint(0, 2**31))


def uint2float(img, dtype_float=np.float32):
    any2float(img, dtype_float)


def any2float(img, dtype_float=np.float32):
    out = img.astype(dtype_float)
    if img.dtype == np.uint16:
        out /= 65535.
    elif img.dtype == np.uint8:
        out /= 255

    return out



class ToTensor(object):
    def __call__(self, pic):

        return torch.from_numpy(any2float(pic.transpose(2, 0, 1)))

    def __repr__(self):
        return self.__class__.__name__ + '()'







class FilteringCollateFn:
    """
    Callable object doing job of collate_fn like default_collate, but does not
    cast batch items with specified key to torch.Tensor.
    Only adds them to list.
    Supports only key-value format batches
    """

    def __init__(self, *keys):
        """
        :param keys: Keys having values that will not be
            converted to tensor and stacked
        """

        self.keys = keys

    def __call__(self, batch):
        if isinstance(batch[0], collections.Mapping):
            result = {}
            for key in batch[0]:
                items = [d[key] for d in batch]
                if key not in self.keys:
                    items = default_collate(items)
                result[key] = items
            return result
        else:
            return default_collate(batch)



def get_part_data(args, part):
    if part == 'test':
        if 'test_csv' in vars(args) and args.test_csv != "":
            test_csv_path  = args.test_csv
        else:
            test_csv_path = f"{args.splits_dir}/test.csv"

        return pd.read_csv(test_csv_path)

    else:
        return pd.read_csv(f"{args.splits_dir}/{part}.csv")

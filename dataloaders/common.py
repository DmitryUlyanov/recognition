from PIL import Image
import cv2
import random 
import numpy as np 
from huepy import red 
import torch

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

def get_image_pil(path):
    img = Image.open(path).convert('RGB')
    return img

def inin_w(worker_id):
    np.random.seed(random.randint(0, 2**31))

def uint2float(img):
    out = img.astype(np.float32)
    if img.dtype == np.uint16:
        out /= 65535.
    elif img.dtype == np.uint8:
        out /= 255

    return out

class ToTensor16(object):
    def __call__(self, pic):
        return torch.from_numpy(uint2float(pic.transpose(2, 0, 1)).astype(np.float32))

    def __repr__(self):
        return self.__class__.__name__ + '()'
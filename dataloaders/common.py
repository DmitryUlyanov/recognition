from PIL import Image
import cv2
import random 
import numpy as np 
from huepy import red 
from torch.utils.data.sampler import Sampler
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


class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, num_batches, batch_size, proportions=None, replacement=False):
        # super().__init__(data_source)
        self.labels = labels
        self.replacement = replacement
        self.num_batches = num_batches
        self.batch_size = batch_size


        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)

        # fractions = [0.95, 0.05]
        if proportions is None:
            self.class_per_batch = {cls: int(self.batch_size / self.num_classes) for cls in self.classes}
        else:
            self.class_per_batch = {cls: int(self.batch_size * proportions[i]) for i, cls in enumerate(self.classes)}


        print('P', self.class_per_batch, self.num_classes, self.batch_size, self.replacement)

        self.class2indices = {cls: np.where(self.labels == cls)[0] for cls in self.classes}
        

    def __iter__(self):
        for i in range(self.num_batches):
            batch = []

            for cls in self.class2indices:
                batch.extend(np.random.choice(self.class2indices[cls], self.class_per_batch[cls], replace=self.replacement))
    
            
            if len(batch) < self.batch_size:
                batch = list(batch) + list(np.random.choice(np.arange(len(self.labels)), self.batch_size - len(batch)))
            
            batch = np.random.permutation(batch)

            yield batch

    def __len__(self):
        return self.num_batches


class ToTensor16(object):
    def __call__(self, pic):
        return torch.from_numpy(uint2float(pic.transpose(2, 0, 1)).astype(np.float32))

    def __repr__(self):
        return self.__class__.__name__ + '()'
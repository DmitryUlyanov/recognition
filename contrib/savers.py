from pathlib import Path
import math
import torch
from torch.optim.optimizer import Optimizer
from utils.utils import parse_dict
from utils.task_queue import TaskQueue
import shutil
from huepy import red
import torch
import sys
import os 
import numpy as np
import copy
import cv2 


def get_saver(name, saver_args=''):
    if name in sys.modules[__name__].__dict__:
        return sys.modules[__name__].__dict__[name](**parse_dict(saver_args))
    else:
        assert False, red(f"Cannot find saver with name {name}")


def npz_per_batch(data, save_dir, iteration):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys
    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    data = tensor_to_np_recursive(copy.deepcopy(data))
    # print(data)
    path = f'{save_dir}/{iteration}.npz'

    # data['labels'] = {'1': data['labels'][0], '2': data['labels'][1]}
    
    np.savez_compressed(path, **data)

    data=None

class Saver(object):
    
    def __init__(self, save_dir, save_fn='npz_per_batch', tq_maxsize = 5, clean_dir=True, num_workers=5):
        super(Saver, self).__init__()
        self.save_dir = Path(str(save_dir))
        self.need_save = True


        if clean_dir and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir) 


        os.makedirs(self.save_dir, exist_ok=True)

        self.tq = TaskQueue(maxsize=tq_maxsize, num_workers=num_workers, verbosity=0) 

        self.save_fn = sys.modules[__name__].__dict__[save_fn]
        # self.need_save = True

    def maybe_save(self, iteration, **kwargs):
        if self.need_save:
            self.tq.add_task(self.save_fn, kwargs, save_dir=self.save_dir, iteration=iteration)  

    def stop(self):
        if self.need_save:
            self.tq.stop_()


class DummySaver(object):
    
    def __init__(self, *args,  **kwargs):
        super().__init__()
      

    def maybe_save(self, iteration, **kwargs):
        pass

    def stop(self):
        pass




def npz_per_item(data, path, args):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys
    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    np.savez_compressed(path, **data)


def tensor_to_np_recursive(data):

    if isinstance(data, torch.Tensor): 
        return data.detach().cpu().numpy() 
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = tensor_to_np_recursive(data[k])

        return data

    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = tensor_to_np_recursive(data[i])

        return data
    else:
        return data
        


from PIL import Image 



def img_per_item(data, save_dir, iteration, dtype='uint8'):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys
    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    data = tensor_to_np_recursive(copy.deepcopy(data))
    path = f'{save_dir}/{iteration}.npz'

    for pred, name in zip(data['output'], data['names']):

        if dtype=='uint8':
            img_to_save = np.round(pred.transpose(1, 2, 0) * 255).astype(np.uint8)
        elif dtype=='uint16':
            img_to_save = np.round(pred.transpose(1, 2, 0) * 65536).astype(np.uint16)
        elif dtype=='depth':
            img_to_save = np.round(pred.transpose(1, 2, 0) * 1000).astype(np.uint16)
        else:
            assert False

        cv2.imwrite(f'{save_dir}/{os.path.basename(name).split(".")[0]:>04}.png', img_to_save[:, :, ::-1],  [cv2.IMWRITE_PNG_COMPRESSION, 9])


def uint8_img_per_item(data, save_dir, iteration):
    return img_per_item(data, save_dir, iteration, 'uint8')

def uint16_img_per_item(data, save_dir, iteration):
    return img_per_item(data, save_dir, iteration, 'uint16')

def depth_img_per_item(data, save_dir, iteration):
    return img_per_item(data, save_dir, iteration, 'depth')
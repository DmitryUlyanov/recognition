from huepy import cyan
import torch 
import torchvision.utils 
import numpy as  np 
from utils.task_queue import TaskQueue
import os
import shutil
from collections import defaultdict

class Meter:
    def __init__(self):
        super().__init__()
        self.data = defaultdict(list)
        self.accumulated = ['topk']

    def update(self, name, val):
        self.data[name].append(val)

    def get_avg(self, name):
        if name in self.accumulated:
            return self.get_last(name)

        return np.mean(self.data[name])

    def get_last(self, name):
        return self.data[name][-1]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccumulatedMetric(object):
    """Computes and stores the average and current value"""

    def __init__(self, metric_fn):
        self.metric_fn = metric_fn
        self.saved_targets = None
        self.saved_preds = None

        self.val = 0
        self.avg = 0

    def update(self, preds, target):

        if self.saved_preds is None:
            self.saved_targets = target
            self.saved_preds = preds
        else:
            self.saved_targets = torch.cat([self.saved_targets, target], 0)
            self.saved_preds = torch.cat([self.saved_preds, preds], 0)

        self.val = self.metric_fn(preds, target)
        self.avg = self.metric_fn(self.saved_preds, self.saved_targets)


def accuracy(output, target):
    return list(top1(output, target))


def top1(output, target):
    for o, t in zip(output, target):
        yield top1_(o, t)


def top1_(output, target):
    mask = (target != -100)
    mask_sum = mask.sum().item()
    if mask_sum == 0:
        return -1

    _, pred = torch.max(output, 1)

    num_correct = (pred[mask] == target[mask]).sum().item()
    score = num_correct * 100.0 / mask_sum 

    return score



def topk(output, target, k_list=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    res = [list() for i in range(len(k_list))]
    for o, t in zip(output, target):
        for i, score in enumerate(topk_(o, t, k_list)):
            res[i].append(score)

    return res 



def topk_(output, target, k_list):
    mask = (target != -100)
    mask_sum = mask.sum().item()
    if mask_sum == 0:
        for i, k in enumerate(k_list):
            yield -1


    maxk = max(k_list)
    batch_size = t.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    t = target.view(1, -1).expand_as(pred)
    mask = mask.view(1, -1).expand_as(pred)
    correct = pred[mask].eq(t[mask])

    for i, k in enumerate(k_list):
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        yield correct_k.mul_(100.0 / mask_sum).item()


def print_stat(name, now_val, avg_val, num_f=3, color=cyan):
        
    format_str = '{:.%df}' % num_f

    now_str = cyan(format_str.format(now_val))
    avg_str = cyan(format_str.format(avg_val))

    return f'{name} {now_str} ({avg_str})\t'





import matplotlib
import matplotlib.cm

def colorize(value, vmin=0, vmax=1, cmap='viridis'):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    return np.ascontiguousarray(value[:, :, :3].transpose(2, 0, 1))




def resize(imgs, sz=256):
    return torch.nn.functional.interpolate(imgs, size=sz)


def get_grid(*args, sz = 256):
    num_img = min( min([len(x) for x in args]), 4)

    grid = []
    for a in args:
        b = a[:num_img].detach().cpu().float()
        if b.shape[1] == 1:
            grid.append(torch.cat( [ torch.from_numpy(colorize(bb)).float()[None, ...]/255 for bb in b ], dim=0 ))
            # grid.append(torch.cat( [b, b, b], dim=1 ) )
        else: 
            grid.append(b[:, :3])

    # print([x.shape for x in grid ])
    imgs = resize( torch.cat(grid), sz=sz)
    x = torchvision.utils.make_grid(imgs, nrow = num_img)
    
    return x





def tensor_to_device_recursive(data):

    if isinstance(data, torch.Tensor): 
        return data.to('cuda', non_blocking=True)
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = tensor_to_device_recursive(data[k])

        return data

    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = tensor_to_device_recursive(data[i])

        return data
    else:
        return data








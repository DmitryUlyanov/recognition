from huepy import cyan
import torch 
import torchvision.utils 
import numpy as  np 
from utils.task_queue import TaskQueue
import os
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









def resize(imgs, sz=256):
    return torch.nn.functional.interpolate(imgs, size=sz)


def get_grid(*args, sz = 256):
    num_img = min( min([len(x) for x in args]), 4)

    grid = []
    for a in args:
        b = a[:num_img].detach().cpu()
        if b.shape[1] == 1:
            grid.append(torch.cat( [b, b, b], dim=1 ) )
        else: 
            grid.append(b[:, :3])

    # print([x.shape for x in grid ])
    imgs = resize( torch.cat(grid), sz=sz)
    x = torchvision.utils.make_grid(imgs, nrow = num_img)
    
    return x



class Saver(object):
    
    def __init__(self, args, save_fn, tq_maxsize = 5, clean_dir=True, num_workers=5):
        super(Saver, self).__init__()
        self.args = args

        self.save_dir = args.dump_path
        self.need_save = False
        if 'save_driver' in args and args.save_driver is not None:
            
            # print('-----------------')
            if clean_dir and os.path.exists(args.dump_path):
                import shutil
                shutil.rmtree(args.dump_path) 

            os.makedirs(args.dump_path, exist_ok=True)

            self.tq = TaskQueue(maxsize=args.batch_size * 2, num_workers=num_workers, verbosity=0) 

            self.save_fn = save_fn
            self.need_save = True

    def maybe_save(self, iteration, **kwargs):
        if self.need_save:
            self.tq.add_task(self.save_fn, kwargs, save_dir=self.save_dir, args=self.args, iteration=iteration)  

    def stop(self):
        if self.need_save:
            self.tq.stop_()



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

def npz_per_batch(data, save_dir, args, iteration):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys
    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    data = tensor_to_np_recursive(data)
    path = f'{save_dir}/{iteration}.npz'

    np.savez_compressed(path, **data)

    data=None


from collections import defaultdict
class Meter:
    def __init__(self):
        super().__init__()
        self.data = defaultdict(list)
        self.accumulated = ['topk']
    # def __getitem__(self, val):


    def update(self, name, val):
        self.data[name].append(val)

    def get_avg(self, name):
        if name in self.accumulated:
            return self.get_last(name)

        return np.mean(self.data[name])

    def get_last(self, name):
        return self.data[name][-1]

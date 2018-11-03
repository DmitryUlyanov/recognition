from huepy import cyan
import torch 
import torchvision.utils 

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


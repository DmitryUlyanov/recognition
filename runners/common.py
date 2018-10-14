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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    return accuracy_(output, target, topk=(1,))[0]

def accuracy_(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    res = [list() for i in range(len(topk))]
    for o, t in zip(output, target):
        mask = (t != -100)
        mask_sum = mask.sum().item()
        if mask_sum == 0:
            for i, k in enumerate(topk):
                res[i].append(-1)

            continue

        maxk = max(topk)
        batch_size = t.size(0)

        _, pred = o.topk(maxk, 1, True, True)
        pred = pred.t()

        # print(pred.shape, t.shape)
        t = t.view(1, -1).expand_as(pred)
        mask = mask.view(1, -1).expand_as(pred)
        correct = pred[mask].eq(t[mask])

        for i, k in enumerate(topk):
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res[i].append(correct_k.mul_(100.0 / mask_sum).item())

    return res 

def print_stat(name, now_val, avg_val, num_f=3, color=cyan):
        
    format_str = '{:.%df}' % num_f

    now_str = cyan(format_str.format(now_val))
    avg_str = cyan(format_str.format(avg_val))

    return f'{name} {now_str} ({avg_str})\t'









def resize(imgs, sz=256):
    return torch.nn.functional.interpolate(imgs, size=sz)

def get_grid(x, y, output):
    num_img = min(x.shape[0], 4)
    imgs = resize(torch.cat([x[:num_img].detach().cpu(), y[:num_img].detach().cpu(), output[:num_img].detach().cpu()]))
    x = torchvision.utils.make_grid(imgs, nrow = num_img)
    
    return x


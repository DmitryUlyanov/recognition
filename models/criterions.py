from tqdm import tqdm
from huepy import red 
import numpy as np
import torch

from torch import nn
import torch.nn.functional as fnn
from torch.nn.modules.loss import _Loss
import sys 

def get_loss(name, **kwargs):
    if name in sys.modules[__name__].__dict__:
        return sys.modules[__name__].__dict__[name](**kwargs)
    elif name in torch.nn.modules.__dict__:
        return torch.nn.modules.__dict__[name](**kwargs)
    else:
        assert False, red(f"Cannot find loss with name {name}")

# -------------------------------------------------
# ----------------- Lap1Loss  ---------------------
# -------------------------------------------------

class LapL1Loss(_Loss):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapL1Loss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp( (x - size // 2)**2 / (-2 * sigma**2) )**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :], requires_grad=False)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


# -------------------------------------------------
# ------------- L1CosineLoss  ---------------
# -------------------------------------------------

class L1CosineLoss(_Loss):

    def __init__(self, l=1):
        super(L1CosineLoss, self).__init__()
        self.l = l

    def __call__(self, input, target):
        return torch.nn.functional.l1_loss(input, target) + 1 - (self.l * torch.nn.functional.cosine_similarity(input, target, dim=1)).mean()


    def cuda(self):
        return self



# -------------------------------------------------
# ------------- MultiHeadCriterion  ---------------
# -------------------------------------------------



class MultiHeadCriterion(_Loss):
    '''
        For a number of "1-of-K" tasks. 

    '''
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MultiHeadCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        '''
            let N be a number of different tasks 

            `target` is a list of size N, where each element is either a vector of size `B` (then is is multi-class task)
                or of size `B` x P_i, where P_i is the number of labels (in multi-label task e.g. tagging)

            `target` is a list of size N, where each element is of size B x P_i for i = 1 ... N
        '''

        losses = []
        for inp, tar in zip(input, target):
            if tar is None:
                continue

            if len(tar.shape) == 1 or tar.shape[1] == 1:
                loss = nn.CrossEntropyLoss()
                losses.append(loss(inp, tar))
            elif tar.shape[1] == inp.shape[1]:
                loss = nn.BCEWithLogitsLoss()

                losses.append(loss(inp, tar))

        loss = sum(losses) / len(losses)

        return loss



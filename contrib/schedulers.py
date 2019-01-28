import math
import torch
from torch.optim.optimizer import Optimizer

from huepy import red
import torch
import sys

def get_scheduler(name):
    if name in sys.modules[__name__].__dict__:
        return sys.modules[__name__].__dict__[name]
    elif name in torch.optim.lr_scheduler.__dict__:
        return torch.optim.lr_scheduler.__dict__[name]
    else:
        assert False, red(f"Cannot find scheduler with name {name}")

import torch.nn as nn

class NoParam(nn.Module):
    def __init__(self, model):
        super(NoParam, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def parameters(self):
        for param in []:
            yield param

    def named_parameters(self, memo=None, prefix=''):
        for param in []:
            yield param
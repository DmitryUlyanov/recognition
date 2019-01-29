import torch
import torch.nn as nn
import numpy as np

def set_param_grad(model, value, set_eval_mode=True):
    for param in model.parameters():
        param.requires_grad = value
    
    if set_eval_mode:
        model.eval()


class MultiHead(nn.Module):
    def __init__(self, in_features=None, num_classes=None):
        super(MultiHead, self).__init__()

        heads = [torch.nn.Linear(in_features, x) for x in num_classes]
        self.heads = torch.nn.ModuleList(heads)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        return [head(input) for head in self.heads]


# class NoParam(nn.Module):
#     def __init__(self, model):
#         super(NoParam, self).__init__()
#         self.model = model

#     def forward(self, x):
#         return self.model(x)

#     def parameters(self):
#         for param in []:
#             yield param

#     def named_parameters(self, memo=None, prefix=''):
#         for param in []:
#             yield param

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)



def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def norm(num_features, tp='bn'):
    if tp == 'bn':
        return nn.BatchNorm2d(num_features)
    elif tp == 'in':
        return nn.InstanceNorm2d(num_features)
    elif tp == 'none':
        return Identity()


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride', conv_class=nn.Conv2d):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection' and to_pad != 0:
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = conv_class(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    # print (layers22)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    # print (layers)
    return nn.Sequential(*layers)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)



class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)




class BaseModel(torch.nn.Module):
    def __init__(self, feature_extractor, predictor, pooling='avg'):
        super(BaseModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.predictor = predictor

        if pooling == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pooling == 'concat':
            self.pool = torch.nn.AdaptiveConcatPool2d((1, 1))
        else:
            raise False


    def forward(self, input):
        x = self.feature_extractor(input)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.predictor(x)

        return x
        


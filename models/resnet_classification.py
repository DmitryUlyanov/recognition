import torch
import torch.nn as nn
import torchvision.models as models
import os
from .model import get_abstract_net, get_model_args
from .criterions import MultiHeadCriterion
import torchvision.transforms as transforms
from .common import NoParam

@get_model_args
def get_args(parser):
    parser.add('--dropout_p',     type=float,  default=0.5,)
    parser.add('--arch',          type=str,    default='resnet18')
    parser.add('--checkpoint',    type=str,    default="")
    parser.add('--n_classes',     type=str,    default="")

    parser.add('--layers_to_fix', type=str, default="")

    return parser


@get_abstract_net
def get_net(args):
    
    load_pretrained = (args.net_init == 'pretrained') and (args.checkpoint == '')
    if load_pretrained:
        print('Loading a net, pretrained on ImageNet1k.')

    model = models.__dict__[args.arch](pretrained=load_pretrained)

    # Hack to make it work with any image size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    if args.layers_to_fix != '':
        for l in args.layers_to_fix.split(','):
            model[l] = NoParam(model[l])

    # if args.use_cond:
    #     conv1_ = model.conv1
    #     model.conv1 = torch.nn.Conv2d(conv1_.in_channels * 3, conv1_.out_channels, kernel_size=conv1_.kernel_size, stride=conv1_.stride, padding=conv1_.padding, bias=False)
    #     model.conv1.weight.data[:, 0:3] = conv1_.weight.data/3
    #     model.conv1.weight.data[:, 3:6] = conv1_.weight.data/3
    #     model.conv1.weight.data[:, 6:9] = conv1_.weight.data/3

    # TableModule(model.modules[0], 3, 1)

    model = MultiHead(model, args)

    return model

# def get_default_criterion():
#     return MultiHeadCriterion()

def get_native_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class MultiHead(nn.Module):
    def __init__(self, main, args):
        super(MultiHead, self).__init__()
        self.main = main

        heads = [torch.nn.Linear(main.fc.in_features, int(x)) for x in args.n_classes.split(',')]
        self.main.fc = nn.Sequential(nn.Dropout(args.dropout_p))
        self.heads = torch.nn.ModuleList(heads)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        input = self.main(input)
        return [head(input) for head in self.heads]


class TableModule(nn.Module):
    def __init__(self, layer, n_chunks, dim):
        super(TableModule, self).__init__()
        
        self.n_chunks = n_chunks
        self.dim = dim
        self.layer = layer

    def forward(self, input, dim):
        chunks = x.chunk(self.n_chunks, self.dim)
        y = torch.cat([self.layer(x) for x in chunks], self.dim)

        return y
    


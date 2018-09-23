import torch
import torch.nn as nn
import torchvision.models as models
import os
from .model import get_abstract_net, get_model_args
import torchvision.transforms as transforms
from torch.nn.modules.loss import _Loss

@get_model_args
def get_args(parser):
    parser.add('--dropout_p', default=0.5, type=float)
    parser.add('--arch', default='resnet18', type=str)
    # parser.add('--not_pretrained', default=False, action='store_true')
    parser.add('--checkpoint', type=str, default="")
    parser.add('--n_classes', type=str, default="")


    return parser


@get_abstract_net
def get_net(args):
    
    load_pretrained = (args.net_init == 'pretrained') and (args.checkpoint == '')
    model = models.__dict__[args.arch](pretrained=load_pretrained)

    # Hack to make it work with any image size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # if args.use_cond:
    #     conv1_ = model.conv1
    #     model.conv1 = torch.nn.Conv2d(conv1_.in_channels * 3, conv1_.out_channels, kernel_size=conv1_.kernel_size, stride=conv1_.stride, padding=conv1_.padding, bias=False)
    #     model.conv1.weight.data[:, 0:3] = conv1_.weight.data/3
    #     model.conv1.weight.data[:, 3:6] = conv1_.weight.data/3
    #     model.conv1.weight.data[:, 6:9] = conv1_.weight.data/3

    # TableModule(model.modules[0], 3, 1)

    model = MultiHead(model, args)
    criterion = MultiHeadCriterion()

    return model, criterion

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
            if len(tar.shape) == 1 or tar.shape[1] == 1:
                loss = nn.CrossEntropyLoss()
                print('CrossEntropy')
                losses.append(loss(inp, tar))
            elif tar.shape[1] == inp.shape[1]:
                loss = nn.BCEWithLogitsLoss()

                losses.append(loss(inp, tar))

        loss = sum(losses)/len(losses)

        return loss
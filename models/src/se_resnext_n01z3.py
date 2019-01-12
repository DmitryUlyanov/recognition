from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys
from models.model import get_abstract_net, get_model_args, BaseModel
from models.criterions import MultiHeadCriterion
import torchvision.transforms as transforms
from models.common import NoParam, MultiHead
from huepy import yellow 

import os
import torch
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .resnext import ResNeXt
from .se_module import SEBlock


@get_model_args
def get_args(parser):
    parser.add('--dropout_p',     type=float,  default=0.5)
    parser.add('--num_classes',   type=str,    default="")

    return parser


@get_abstract_net
def get_net(args):
    
    load_pretrained = args.net_init == 'pretrained' and args.checkpoint == ""
    if load_pretrained:
        print(yellow('Loading a net, pretrained on ImageNet1k.'))

    model = SE_ResNeXt101FT(num_classes = 340, pretrained=load_pretrained)
    
    if not args.load_only_extractor: 
        return model


    model.load_state_dict(torch.load('extensions/qd/data/experiments/se_resnext_n01z3/checkpoints/se_resnext101_n.pth')['state_dict'])


        # Extractor
    feature_extractor = model.features
     
    # Construct
   


    num_classes = [int(x) for x in args.num_classes.split(',')]

    predictor = nn.Sequential( nn.Dropout(args.dropout_p), MultiHead(in_features = 2048, num_classes=num_classes))
        
    model = BaseModel(feature_extractor, predictor)

    
    return model




def get_native_transform():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])








__all__ = ['se_resnext50', 'se_resnext101', 'se_resnext101_64', 'se_resnext152']

model_urls = {
    'se_resnext50': 'https://nizhib.ai/share/pretrained/se_resnext50-5cc09937.pth'
}


class SEBottleneck(nn.Module):
    """
    SE-RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None,
                 reduction=16):
        super(SEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * 4, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnext50(num_classes=1000, pretrained=False):
    """Constructs a SE-ResNeXt-50 model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['se_resnext50']))
    return model


def se_resnext101(num_classes=1000, pretrained=True):
    """Constructs a SE-ResNeXt-101 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 23, 3], num_classes=num_classes)

    if pretrained:
        path = os.path.join(os.getenv("HOME"), '.torch/models/se_resnext101_best.pth')
        pretrained_parallel_state_dict = torch.load(path, map_location='cpu')['state_dict']
        pretrained_normal_state_dict = dict()
        for key in pretrained_parallel_state_dict.keys():
            # key = key
            pretrained_normal_state_dict[key.split('module.')[1]] = \
            pretrained_parallel_state_dict[key]
        model.load_state_dict(pretrained_normal_state_dict)

    return model


def se_resnext101_64(num_classes=1000):
    """Constructs a SE-ResNeXt-101 (64x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 64, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext152(num_classes=1000):
    """Constructs a SE-ResNeXt-152 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 8, 36, 3], num_classes=num_classes)
    return model


class SE_ResNeXt101FT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        model = se_resnext101(pretrained=pretrained)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return [x]

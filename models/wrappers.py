import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import os
import sys

from huepy import yellow 


from .src import unet
from .src import resnext
from .src import inception_v4

from dataloaders.augmenters import Identity
import torchvision.transforms as transforms
import torchvision
from models.common import MultiHead, BaseModel


class ResNext(object):
    def __init__(self, arg):
        super(UNet, self).__init__()
        self.arg = arg
        
    @staticmethod
    def get_args(parser):
        parser.add('--arch',  choices=['resnext50', 'resnext101', 'resnext101_64', 'resnext152'], default='resnext50')
        parser.add('--pooling',  choices=['avg', 'max', 'concat'], default='avg')


        return parser

    @staticmethod
    def get_net(args):

        load_pretrained = args.net_init == 'pretrained'
        if load_pretrained:
            print(yellow(' - Loading a net, pretrained on ImageNet1k.'))

        model = resnext.__dict__[args.arch](num_classes=1000, pretrained=load_pretrained)


        # Extractor
        feature_extractor = nn.Sequential(
                                    model.conv1,
                                    model.bn1,
                                    model.relu,
                                    model.maxpool,
                                    model.layer1,
                                    model.layer2,
                                    model.layer3,
                                    model.layer4)

         
        # Predictor 
        predictor = MultiHead(in_features = model.fc.in_features, num_classes=[int(x) for x in args.num_classes.split(',')])
        if args.dropout_p > 0:
            predictor = nn.Sequential( nn.Dropout(args.dropout_p), predictor)



        # Construct
        model = BaseModel(feature_extractor, predictor, args.pooling)


        return model








class UNet(object):
    def __init__(self, arg):
        super(UNet, self).__init__()
        self.arg = arg
        
    @staticmethod
    def get_args(parser):
        parser.add('--num_input_channels',  type=int, default=3)
        parser.add('--num_output_channels', type=int, default=3)

        parser.add('--feature_scale', type=int, default=4)
        parser.add('--more_layers',   type=int, default=0)
        parser.add('--concat_x',      default=False, action='store_bool')

        parser.add('--upsample_mode', type=str, default="deconv")
        parser.add('--pad',           type=str, default="zero")

        parser.add('--norm_layer', type=str, default="in")
        
        parser.add('--last_act', default='sigmoid',  type=str)

        return parser

    @staticmethod
    def get_net(args):

        model = unet.UNet(
                     num_input_channels  = args.num_input_channels,
                     num_output_channels = args.num_output_channels,
                     feature_scale       = args.feature_scale, 
                     more_layers         = args.more_layers, 
                     concat_x            = args.concat_x,
                     upsample_mode       = args.upsample_mode, 
                     pad                 = args.pad, 
                     norm_layer          = args.norm_layer, 
                     last_act            = args.last_act, 
                     need_bias           = True
        )

        return model









class InceptionV4(object):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        
    @staticmethod
    def get_args(parser):
        parser.add('--dropout_p',     type=float,  default=0.5)
        parser.add('--num_classes',   type=str,    default="")
        parser.add('--pooling',   choices=['avg', 'max', 'concat'], default='avg')

        return parser

    @staticmethod
    def get_net(args):

        load_pretrained = args.net_init == 'pretrained'
        if load_pretrained:
            print(yellow(' - Loading a net, pretrained on ImageNet1k.'))

        model = inception_v4.InceptionV4(num_classes=1001, pretrained=load_pretrained)

        num_classes = [int(x) for x in args.num_classes.split(',')]


        predictor = MultiHead(in_features = 1536, num_classes=num_classes)
        predictor = nn.Sequential( nn.Dropout(args.dropout_p), predictor)

        
        # Construct
        model = BaseModel(model.features, predictor, args.pooling)


        return model

    @staticmethod
    def get_native_transform():
        return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])



class ResNet(object):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        
    @staticmethod
    def get_args(parser):
        parser.add('--dropout_p',     type=float,  default=0.5)
        parser.add('--arch',          type=str,    default='resnet18')
        parser.add('--num_classes',   type=str,  default="")

        parser.add('--num_input_channels',  type=int,  default=3)

        parser.add('--layers_to_fix', type=str, default="")

        return parser

    @staticmethod
    def get_net(args):

        load_pretrained = args.net_init == 'pretrained'
        if load_pretrained:
            print(yellow(' - Loading a net, pretrained on ImageNet1k.'))

        resnet = torchvision.models.__dict__[args.arch](pretrained=load_pretrained)

        # If an image has different number of channelss
        if args.num_input_channels != 3:
            if args.num_input_channels % 3 != 0:
                assert False

            conv1_ = resnet.conv1
            resnet.conv1 = torch.nn.Conv2d(
                args.num_input_channels, 
                conv1_.out_channels, 
                kernel_size=conv1_.kernel_size, 
                stride=conv1_.stride, 
                padding=conv1_.padding, 
                bias=False
            )

            for i in range(int(args.num_input_channels / 3)):
                resnet.conv1.weight.data[:, i * 3: (i + 1) * 3] = conv1_.weight.data / 3


        # Extractor
        feature_extractor = nn.Sequential(
                                    resnet.conv1,
                                    resnet.bn1,
                                    resnet.relu,
                                    resnet.maxpool,
                                    resnet.layer1,
                                    resnet.layer2,
                                    resnet.layer3,
                                    resnet.layer4)

         
        # Predictor 
        predictor = MultiHead(in_features = resnet.fc.in_features, num_classes=[int(x) for x in args.num_classes.split(',')])
        if args.dropout_p > 0:
            predictor = nn.Sequential( nn.Dropout(args.dropout_p), predictor)

        # Construct
        model = BaseModel(feature_extractor, predictor)

        return model


    @staticmethod
    def get_native_transform():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225])
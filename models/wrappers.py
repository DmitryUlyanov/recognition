import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import os
import sys

from huepy import yellow 


from .src import unet
from dataloaders.augmenters import Identity
import torchvision.transforms as transforms
from models.common import MultiHead


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
        super(UNet, self).__init__()
        self.arg = arg
        
    @staticmethod
    def get_args(parser):
         parser.add('--dropout_p',     type=float,  default=0.5,)
         parser.add('--num_classes',   type=str,    default="")

        return parser

    @staticmethod
    def get_net(args):

        load_pretrained = args.net_init == 'pretrained' and args.checkpoint == ""
        if load_pretrained:
            print(yellow('Loading a net, pretrained on ImageNet1k.'))

        model = InceptionV4(num_classes=1001, pretrained=load_pretrained)

        num_classes = [int(x) for x in args.num_classes.split(',')]

        predictor = MultiHead(in_features = 1536, num_classes=num_classes)
        # if args.dropout_p > 0:
        predictor = nn.Sequential( nn.Dropout(args.dropout_p), predictor)
        
        model.predictor =  predictor

        return model

    @staticmethod
    def get_native_transform():
        return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from models.model import get_abstract_net, get_model_args
import torchvision.transforms as transforms
from models.common import NoParam, MultiHead, set_param_grad, ListModule
from huepy import yellow 

@get_model_args
def get_args(parser):
    parser.add('--dropout_p',     type=float,  default=0.5,)
    parser.add('--num_classes',   type=int,    default=1)
    parser.add('--freeze_basenet',action='store_bool', default=False)

    return parser


@get_abstract_net
def get_net(args):
    
    load_pretrained = args.net_init == 'pretrained' and args.checkpoint != ''
    if load_pretrained:
        print(yellow('Loading a net, pretrained on ImageNet1k.'))

    model = LinkNet(args.num_classes, depth=34, pretrained=load_pretrained)

    if args.freeze_basenet:
        model.freeze_basenet()
    else:
        model.unfreeze_basenet()
    return model


def get_native_transform():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])




class ResNetEncoder(nn.Module):
    def __init__(self, arch, pretrained=True):
        super().__init__()

        backbone = arch(pretrained=pretrained)

        self.encoder0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.filters = [self.encoder1[-1].conv2.out_channels,
                        self.encoder2[-1].conv2.out_channels,
                        self.encoder3[-1].conv2.out_channels,
                        self.encoder4[-1].conv2.out_channels]

    def forward(self, x):
        acts = []
        x = self.encoder0(x)
        x = self.encoder1(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder2(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder3(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder4(x)
        # print(x.shape)
        acts.append(x)
        return acts

import torch
import torch.nn as nn


class GroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

class DecoderBlock(nn.Module):
    def __init__(self, m, n, stride=2):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(m, m // 4, 1)
        self.norm1 = nn.BatchNorm2d(m // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.conv2 = nn.ConvTranspose2d(m // 4, m // 4, 3, stride=stride, padding=1)
        self.norm2 = nn.BatchNorm2d(m // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(m // 4, n, 1)
        self.norm3 = nn.BatchNorm2d(n)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        double_size = (x.size(-2) * 2, x.size(-1) * 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x, output_size=double_size)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, num_filters, num_classes=2):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(num_filters, num_filters // 2, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters // 2, num_filters // 2, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_filters // 2, num_classes, 1)

    def forward(self, inputs):
        double_size = (inputs.size(-2) * 2, inputs.size(-1) * 2)
        x = self.conv1(inputs, output_size=double_size)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class LinkNet(nn.Module):
    def __init__(self, num_classes, depth=18, pretrained=True):
        super().__init__()

        if depth == 18:
            self.encoder = ResNetEncoder(models.resnet18, pretrained=pretrained)
        elif depth == 34:
            self.encoder = ResNetEncoder(models.resnet34, pretrained=pretrained)
        else:
            raise ValueError(f'Unexcpected LinkNet depth: {depth}')
        filters = self.encoder.filters
        # self._recompile_intro()

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final = FinalBlock(filters[0], num_classes)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # d4 = normgrad(self.decoder4(e4)) + normgrad(e3)
        # d3 = normgrad(self.decoder3(d4)) + normgrad(e2)
        # d2 = normgrad(self.decoder2(d3)) + normgrad(e1)
        # d1 = normgrad(self.decoder1(d2))


        return self.final(d1)

    def freeze_basenet(self):
        for m in [self.encoder]:
            set_param_grad(m, False)

    def unfreeze_basenet(self):
        for m in [self.encoder]:
            set_param_grad(m, True)



class NormGrad(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        # print()
        grad_input = grad_output.clone()
        grad_input = torch.nn.functional.normalize(grad_input)

        print(torch.norm(grad_output), torch.norm(grad_input))
        return grad_input

normgrad = NormGrad.apply 

class NormGradModule(nn.Module):
   def __init__(self, in_features):
       super(NormGradModule,self).__init__()
       

   def forward(self, x):
       out = normgrad(x)
       return out 

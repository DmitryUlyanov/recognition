import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
# from models.model import get_abstract_net, get_model_args
# import torchvision.transforms as transforms
from models.common import set_param_grad, ListModule
from huepy import yellow 

# @get_model_args
# def get_args(parser):
#     parser.add('--num_classes',   type=int,    default=1)
#     parser.add('--freeze_basenet',action='store_bool', default=False)

#     return parser


# @get_abstract_net
# def get_net(args):
    
#     # load_pretrained = args.net_init == 'pretrained' and args.checkpoint != ''
#     # if load_pretrained:
#     #     print(yellow('Loading a net, pretrained on ImageNet1k.'))

#     model = LinkNet(args.num_classes, depth=34, pretrained=load_pretrained)

#     if args.freeze_basenet:
#         model.freeze_basenet()
#     else:
#         model.unfreeze_basenet()
#     return model


# def get_native_transform():
#     return transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                  std=[0.5, 0.5, 0.5])




class ResNetEncoder(nn.Module):
    def __init__(self, arch, num_input_channels=3, pretrained=True):
        super().__init__()

        backbone = arch(pretrained=pretrained)

         # If an image has different number of channelss
        if num_input_channels != 3:
           

            conv1_ = backbone.conv1
            backbone.conv1 = torch.nn.Conv2d(
                num_input_channels, 
                conv1_.out_channels, 
                kernel_size=conv1_.kernel_size, 
                stride=conv1_.stride, 
                padding=conv1_.padding, 
                bias = (conv1_.bias is not None)
            )

            if pretrained:

                # scaling_factor = 3. / num_input_channels
                scaling_factor = 1/3

                for i in range(int(num_input_channels / 3)):
                    backbone.conv1.weight.data[:, i * 3: (i + 1) * 3] = conv1_.weight.data * scaling_factor

                    if (conv1_.bias is not None) :
                        backbone.conv1.bias.data[:, i * 3: (i + 1) * 3] = conv1_.bias.data * scaling_factor           

                
                backbone.conv1.weight.data[:, num_input_channels - (num_input_channels % 3): num_input_channels] = conv1_.weight.data[:, :num_input_channels % 3] * scaling_factor

                if  (conv1_.bias is not None) :
                    backbone.conv1.bias.data[:, i * 3: (i + 1) * 3] = conv1_.bias.data * scaling_factor

            # else:



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
    def __init__(self, num_input_channels, 
                       num_output_channels, 
                       depth=18, 
                       pretrained=True):
        super().__init__()

        if depth == 18:
            self.encoder = ResNetEncoder(models.resnet18, num_input_channels=num_input_channels, pretrained=pretrained)
        elif depth == 34:
            self.encoder = ResNetEncoder(models.resnet34, num_input_channels=num_input_channels, pretrained=pretrained)
        # elif depth == 50:
        #     self.encoder = ResNetEncoder(models.resnet50, num_input_channels=num_input_channels, pretrained=pretrained)
        else:
            raise ValueError(f'Unexcpected LinkNet depth: {depth}')
        filters = self.encoder.filters
        # self._recompile_intro()

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final = FinalBlock(filters[0], num_output_channels)

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




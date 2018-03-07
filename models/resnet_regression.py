import torch.nn as nn
import torchvision.models as models


def get_args(parser):
    parser.add('--dropout_p', default=0.5, type=float)
    parser.add('--n_outputs', default=10, type=int)
    parser.add('--not_pretrained', default=False, action='store_true')
    parser.add('--checkpoint', type=str)
    parser.add('--arch', default='resnet18', type=str)

    return parser


def get_net(args):
    model = models.__dict__[args.arch](True, pretrained=not args.not_pretrained)

    # Hack to make it work with any image size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    model.fc = nn.Sequential(nn.Dropout(args.dropout_p),
                             nn.Linear(model.fc.in_features, args.n_outputs))
    criterion = nn.L1Loss()

    return model.cuda(), criterion.cuda()

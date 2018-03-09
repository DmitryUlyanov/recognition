import torch
import torch.nn as nn
import torchvision.models as models
from .common import NoParam

def get_args(parser):
    parser.add('--dropout_p', default=0.5, type=float)
    parser.add('--arch', default='resnet18', type=str)
    parser.add('--not_pretrained', default=False, action='store_true')
    parser.add('--checkpoint', type=str, default="")

    return parser


class MultiHead(nn.Module):
    def __init__(self, main, args):
        super(MultiHead, self).__init__()
        self.main = main

        heads = [torch.nn.Linear(main.fc.in_features, x) for x in args.n_classes]
        self.main.fc = nn.Sequential(nn.Dropout(args.dropout_p))
        self.heads = torch.nn.ModuleList(heads)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        input = self.main(input)
        return [head(input) for head in self.heads]

def get_net(args):
    model = models.__dict__[args.arch](pretrained=not args.not_pretrained)

    model.conv1   = NoParam(model.conv1)
    model.bn1     = NoParam(model.bn1)
    model.relu    = NoParam(model.relu)
    model.layer1  = NoParam(model.layer1)
    model.layer2  = NoParam(model.layer2)
    # model.layer3  = NoParam(model.layer3)
    # model.layer4  = NoParam(model.layer4)
    # model.layer3  = NoParam(model.layer3)

    
    
    # Hack to make it work with any image size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    model = MultiHead(model, args)
    criterion = nn.CrossEntropyLoss()

    print (model)
    
    if args.checkpoint != '':
        print("Loading pretrained net")
        model.load_state_dict(torch.load(args.checkpoint))
        

    return model.cuda(), criterion.cuda()

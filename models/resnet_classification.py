import torch
import torch.nn as nn
import torchvision.models as models


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
    load_pretrained = (not args.not_pretrained) and (args.checkpoint == '')
    
    model = models.__dict__[args.arch](pretrained=load_pretrained)

    # Hack to make it work with any image size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    model = MultiHead(model, args)
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint != '':
        print("Loading pretrained net")
        model.load_state_dict(torch.load(args.checkpoint))
        

    return model.cuda(), criterion.cuda()

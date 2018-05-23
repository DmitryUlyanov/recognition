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
    

# input = Variable(torch.rand(2, 5))
# net = TableModule()
# output = net(input, 1)


def get_net(args):
    load_pretrained = (not args.not_pretrained) and (args.checkpoint == '')
    
    model = models.__dict__[args.arch](pretrained=load_pretrained)

    # Hack to make it work with any image size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    if args.use_cond:
        conv1_ = model.conv1
        model.conv1 = torch.nn.Conv2d(conv1_.in_channels * 3, conv1_.out_channels, kernel_size=conv1_.kernel_size, stride=conv1_.stride, padding=conv1_.padding, bias=False)
        model.conv1.weight.data[:, 0:3] = conv1_.weight.data/3
        model.conv1.weight.data[:, 3:6] = conv1_.weight.data/3
        model.conv1.weight.data[:, 6:9] = conv1_.weight.data/3

    # TableModule(model.modules[0], 3, 1)

    model = MultiHead(model, args)
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint != '':
        print("Loading pretrained net")
        model.load_state_dict(torch.load(args.checkpoint))
        

    return model.cuda(), criterion.cuda()

import torch.nn as nn
import torchvision.models as models


def get_args(parser):
    parser.add('--dropout_p', default=0.5, type=float)
    parser.add('--n_outputs', default=10, type=int)

    return parser


def get_net(args):
    model = models.resnet34(True)
    model.fc = nn.Sequential(nn.Dropout(args.dropout_p),
                             nn.Linear(model.fc.in_features, args.n_outputs))
    criterion = nn.L1Loss()

    return model.cuda(), criterion.cuda()

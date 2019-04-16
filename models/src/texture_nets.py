import torch
import torch.nn as nn
from .common import * 


normalization = nn.BatchNorm2d


# def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
#     if pad == 'zero':
#         return nn.Conv2d(in_f, out_f, kernel_size, stride, padding=(kernel_size - 1) / 2, bias=bias)
#     elif pad == 'reflection':
#         layers = [nn.ReflectionPad2d((kernel_size - 1) / 2),
#                   nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)]
#         return nn.Sequential(*layers)

def get_texture_nets(inp=3, 
                     ratios = [32, 16, 8, 4, 2, 1], 
                     fill_noise=False, pad='zero', 
                     need_sigmoid=False, 
                     conv_num=8, 
                     upsample_mode='nearest'):


    for i in range(len(ratios)):
        j = i + 1

        seq = nn.Sequential()

        tmp =  nn.AvgPool2d(ratios[i], ratios[i])

        seq.add(tmp)
        if fill_noise:
            seq.add(GenNoise(inp))

        seq.add(conv(inp, conv_num, 3, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        seq.add(conv(conv_num, conv_num, 5, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        seq.add(conv(conv_num, conv_num, 3, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        if i == 0:
            seq.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            cur = seq
        else:

            cur_temp = cur

            cur = nn.Sequential()

            # Batch norm before merging 
            seq.add(normalization(conv_num))
            cur_temp.add(normalization(conv_num * (j - 1)))

            cur.add(Concat(1, cur_temp, seq))

            cur.add(conv(conv_num * j, conv_num * j, 3, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            cur.add(conv(conv_num * j, conv_num * j, 5, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            cur.add(conv(conv_num * j, conv_num * j, 7, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            if i == len(ratios) - 1: 
                cur.add(conv(conv_num * j, 3, 1, pad=pad))
                pass
            else:
                cur.add(nn.Upsample(scale_factor=2, mode=upsample_mode)) 
            
    model = cur
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def get_args(parser):
    # parser.add('--dropout_p', default=0.5, type=float)
    # parser.add('--arch', default='resnet18', type=str)
    # parser.add('--not_pretrained', default=False, action='store_true')
    parser.add('--checkpoint', type=str, default="")

    parser.add('--num_input_channels', type=int, default=3)
    parser.add('--num_output_channels', type=int, default=3)

    # parser.add('--feature_scale', type=int, default=4)
    # parser.add('--more_layers', type=int, default=0)
    # parser.add('--concat_x', default=False, action='store_true')

    parser.add('--upsample_mode', type=str, default="deconv")
    parser.add('--pad', type=str, default="zero")

    # parser.add('--norm_layer', type=str, default="in")
    
    # parser.add('--need_sigmoid', default=False, action='store_true')
    # parser.add('--need_bias',    default=T, action='store_true')

    return parser

def get_net(args):
    # load_pretrained = (not args.not_pretrained) and (args.checkpoint == '')
    
    # model = models.__dict__[args.arch](pretrained=load_pretrained)

    # Hack to make it work with any image size
    # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    model = get_texture_nets(inp= args.num_input_channels, 
                             ratios = [32, 16, 8, 4, 2, 1], 
                             fill_noise=False, 
                             pad=args.pad, 
                             need_sigmoid=True, 
                             conv_num=16, 
                             upsample_mode=args.upsample_mode)

    criterion = nn.L1Loss()

    print(model)
    if args.checkpoint != '':
        print("Loading pretrained net")
        # print(model)
        model.load_state_dict(torch.load(args.checkpoint))
        
    return model.cuda(), criterion.cuda()

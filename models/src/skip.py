import torch
import torch.nn as nn
from .common import *

def bn(num_features):
    return nn.BatchNorm2d(num_features)


def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
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

    model = skip(num_input_channels=args.num_input_channels, 
                             num_output_channels=args.num_output_channels, 
                             num_channels_down=[16, 32, 64, 128, 128], 
                             num_channels_up=[16, 32, 64, 128, 128], 
                             num_channels_skip=[4, 4, 4, 4, 4], 
                             filter_size_down=3, 
                             filter_size_up=3, 
                             filter_skip_size=1,
                             need_sigmoid=True, 
                             need_bias=True, 
                             pad='zero', 
                             upsample_mode='nearest', 
                             downsample_mode='stride', 
                             act_fun='LeakyReLU', 
                             need1x1_up=True)

    criterion = nn.L1Loss()

    print(model)
    if args.checkpoint != '':
        print("Loading pretrained net")
        # print(model)
        model.load_state_dict(torch.load(args.checkpoint))
        
    return model.cuda(), criterion.cuda()

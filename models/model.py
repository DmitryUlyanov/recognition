import torch
from models.LSUV import LSUVinit
from huepy import red, yellow, orange

from munch import munchify
from models.common import set_param_grad
from dataloaders.augmenters import Identity
import gzip
from pathlib import Path

import models.wrappers
import os 
from utils.utils import load_module_, load_module


def get_wrapper(model_name, extension):
    if model_name in models.wrappers.__dict__:
        return models.wrappers.__dict__[model_name]
    else:

        m = load_module_(extension, 'models', 'wrappers', raise_error=False)
        # print(m )
        if m is None:
            m = load_module_(extension, 'models', model_name)


        return m.__dict__[model_name]

class Model:
    def __init__(self, model_name, extension):
        # self.model_name = model_name
        self.net_wrapper = get_wrapper(model_name, extension)
        
    # def find_definition(self, model_name, extension):
    #     if model_name in models.wrappers.__dict__:
    #         return models.wrappers.__dict__[model_name]
    #     else:

    #         m = load_module_(extension, 'models', 'wrappers', raise_error=False)

    #         if m is None:
    #             m = load_module_(extension, 'models', model_name)


    #         return m.__dict__[model_name]

    def get_args(self, parser):
        parser.add('--checkpoint',      type=Path)
        parser.add('--net_init',        type=str, default="default", help='pretrained|lsuv|default')
        parser.add('--lsuv_batch_size', type=int, default=-1)

        parser.add('--use_all_gpus',       default=False, action='store_bool')
        
        parser.add('--fix_feature_extractor',    default=False, action='store_bool')
        parser.add('--freeze_bn',                default=False, action='store_bool')
        parser.add('--merge_model_and_loss',     default=False, action='store_bool')

        parser.add('--fp16',                     default=False, action='store_bool')
        parser.add('--compress_checkpoints',     default=False, action='store_bool')
        parser.add('--save_optimizer',           default=False, action='store_bool')

        parser.add('--bn_momentum',  default=-1, type=float)

        parser.add('--checkpoint_strict_load_state',   default=True, action='store_bool')
        parser.add('--checkpoint_load_only_extractor', default=False, action='store_bool')

        parser.add('--fancy_stuff',                    default=True, action='store_bool')

        return self.net_wrapper.get_args(parser)


    def init_weights(self, args, model, train_dataloader):

        print(f" - Intializing weights: {yellow(args.net_init)}")
        if args.net_init == 'lsuv':
            data = iter(train_dataloader).next()['input'].to(args.device)

            if args.lsuv_batch_size > 0:
                data = data[:args.lsuv_batch_size]

            model = LSUVinit(model, data, needed_std=1.0, std_tol=0.1,
                             max_attempts=10, do_orthonorm=False, device=args.device)

        return model

    def get_checkpoint_path(self, args):
        options = [args.checkpoint, args.experiment_dir / 'checkpoints' / args.checkpoint]

        for path in options:
            if path.exists():
                return path
        else:
            assert False, red('Checkpoint path was set, but not found. \n'  + str(options))


    def load_state(self, args, checkpoint_path, model):
        '''
            Returns a model loaded from the checkpoint 
        '''

        print(f" - Loading {yellow(args.model)} from checkpoint {yellow(checkpoint_path)}")
        
        state_dict = torch.load(checkpoint_path)
        if isinstance(state_dict, dict): 
            state_dict = state_dict['state_dict']
        
        # print(state_dict)
        # if 
        # state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model')}
        # state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model')}

        if args.checkpoint_load_only_extractor:
            print(orange(f' !!! Removing predictor weights from the loaded state.'))
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('predictor')}

        model.load_state_dict(state_dict, strict=args.checkpoint_strict_load_state)

        return model


    def get_net(self, args, train_dataloader, criterion):
        
        model = self.net_wrapper.get_net(args)
        model = model.to(args.device)

        if args.fp16:
            model = FP16Model(model)


        # Load checkpoint 
        if args.checkpoint is not None and args.checkpoint != Path('.'):
            
            checkpoint_path = self.get_checkpoint_path(args)
            model = self.load_state(args, checkpoint_path, model)
        else:
            model = self.init_weights(args, model, train_dataloader)
      
        if not args.fancy_stuff:
            return model

        # Some other stuff
        if hasattr(model, 'feature_extractor'):
            value = not args.fix_feature_extractor
            set_param_grad(model.feature_extractor, value = value, set_eval_mode = False)

        if args.freeze_bn:
            print(yellow(' - Freezing BN'))

            def freeze_bn(m):
                if isinstance(m, torch.nn.BatchNorm2d):

                    m.training = False
                    def nop(*args, **kwargs):
                        pass

                    m.train = nop

            model.apply(freeze_bn)

        
        if args.merge_model_and_loss:
            model = ModelAndLoss(model, criterion)

        if args.use_all_gpus and args.device == 'cuda' and torch.cuda.device_count() > 1:
            print(yellow(' - Using all GPU\'s!'))

            model = torch.nn.DataParallel(model) 
          

        if args.bn_momentum != -1:
            def freeze_bn1(m):
                if isinstance(m, torch.nn.BatchNorm2d):

                    m.momentum = args.bn_momentum
                    
                    # def nop(*args, **kwargs):
                    #     pass

                    # m.train = nop

            model.apply(freeze_bn1)


        return model



    def get_native_transform(self):
        if hasattr(self.net_wrapper, 'native_transform'):
            return net_wrapper.native_transform()
        else:
            return Identity

    

def save_model(model, epoch, args, optimizer=None, stage_num=0):
    
    model_to_save = unwrap_model(model)
    # if isinstance(model, torch.nn.DataParallel):
    #     model_to_save = model.module
    
    # if 'ModelAndLoss' in str(type(model_to_save)):
    #     model_to_save = model_to_save.model

    # if 'FP16Model' in str(type(model_to_save)):
    #     model_to_save = model_to_save.network

    dict_to_save = { 
        'state_dict': model_to_save.state_dict(), 
        'args': args
    }

    if args.save_optimizer and optimizer is not None: 
        dict_to_save['optimizer'] = optimizer.state_dict()


    # Dump
    save_path = f'{args.experiment_dir}/checkpoints/stage_{stage_num}_model_{epoch}.pth'

    if args.compress_checkpoints:
        with gzip.open(f'{save_path}.gz', 'wb') as f:
            torch.save(dict_to_save, f, pickle_protocol=-1)

    else:
        torch.save(dict_to_save, save_path, pickle_protocol=-1)






from pathlib import Path

def load_model_from_checkpoint(checkpoint_path, args_to_update=None):

        saved_args = vars(torch.load(checkpoint_path)['args'])


        from utils.argparse_utils import MyArgumentParser
        parser = MyArgumentParser(conflict_handler='resolve')

        # args, default_args, m = get_args_and_modules(parser, phase='test', saved_args=saved_args)



        m_model = load_module(saved_args['extension'], 'models', saved_args['model'])
        m_model.get_args(parser)


        args = vars(parser.parse_args([]))
        args.update(saved_args)

        if args_to_update is not None:
            args.update(args_to_update)

        args = munchify(args)

        # Update
        args.checkpoint = Path(checkpoint_path)
        args.net_init   = 'checkpoint'
        
        args.checkpoint_load_only_extractor = False
        args.checkpoint_strict_load_state = True

        args.use_all_gpus = False
        args.fancy_stuff  = False

        if args_to_update is not None:
            args.update(args_to_update)

        model = m_model.get_net(args, None, None)

        return model, m_model, args




class ModelAndLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelAndLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, target, *data):
        output = self.model(*data)
        loss = self.loss(output, target)

        return loss, output




import apex

class FP16Model(torch.nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network, convert_bn=False):
        super(FP16Model, self).__init__()

        

        self.network = convert_network(network, dtype=torch.half, convert_bn=convert_bn)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)

def convert_network(network, dtype, convert_bn):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if not convert_bn and isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        apex.fp16_utils.convert_module(module, dtype)

    return network





def unwrap_model(model):


    model_ = model
    while True: 
        if isinstance(model_, torch.nn.DataParallel):
            model_ = model_.module

        elif 'ModelAndLoss' in str(type(model_)):
            model_ = model_.model

        elif 'FP16Model' in str(type(model_)):
            model_ = model_.network

        else:
            return model_




import numpy as np


# if state_dict['feature_extractor.0.weight'].shape[1] != args.num_input_channels:
#     print('Surgery ==============')
#     t = torch.zeros( (state_dict['feature_extractor.0.weight'].shape[0], args.num_input_channels, state_dict['feature_extractor.0.weight'].shape[2], state_dict['feature_extractor.0.weight'].shape[3]), dtype=torch.float)

#     for i in range(int(args.num_input_channels / 3)):
#         t[:, i * 3: (i + 1) * 3] = state_dict['feature_extractor.0.weight'] / (int(args.num_input_channels / 3))

#     if args.num_input_channels % 3 > 0: 
#         t[:, - (args.num_input_channels % 3) : ] = state_dict['feature_extractor.0.weight'][:, -(args.num_input_channels % 3) :]


#     state_dict['feature_extractor.0.weight'] = t
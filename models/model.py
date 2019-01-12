import torch
from models.LSUV import LSUVinit
from huepy import red, yellow
from utils.utils import load_module
from munch import munchify
from models.common import set_param_grad
from dataloaders.augmenters import Identity
import gzip
from pathlib import Path

import models.wrappers
import os 

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.net_wrapper = self.find_wrapper()

    def find_module(self):
        if model_name in models.wrappers.__dict__:
            return models.wrappers[model_name]
        else:
            assert False, 'Cannot find model wrapper'
            

    def get_model_args(self, get_args):
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
                checkpoint_path = args.checkpoint
                break
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
        
        if args.checkpoint_load_only_extractor:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('predictor')}

        model.load_state_dict(state_dict, strict=args.checkpoint_strict_load_state)

        return model


    def get_net(self, args, train_dataloader, criterion):
        
        model = self.net_wrapper.get_net(args)
        model = model.to(args.device)

        if args.fp16:
            model = model.half()


        # Load checkpoint 
        if args.checkpoint is not None:
            
            checkpoint_path = self.get_checkpoint_path()
            model = self.load_state(args, checkpoint_path, model)
        else:
            model = self.init_weights(args, model, train_dataloader)
      
        
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



    def get_native_transform():
        if hasattr(self.net_wrapper, 'native_transform'):
            return net_wrapper.native_transform()
        else:
            return Identity

    return wrapper

def save_model(model, epoch, args, optimizer=None, stage_num=0):
    
    model_to_save = model
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module

    if 'ModelAndLoss' in str(type(model_to_save)):
        model_to_save = model_to_save.model

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

    torch.save(dict_to_save, save_path, pickle_protocol=-1)





class BaseModel(torch.nn.Module):
    def __init__(self, feature_extractor, predictor):
        super(BaseModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = self.feature_extractor(input)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.predictor(x)

        return x
        


def load_model_from_checkpoint(checkpoint_path, args_to_update=None):

        args = vars(torch.load(checkpoint_path)['args'])

        args['checkpoint'] = checkpoint_path
        args['net_init'] = 'checkpoint'
        args['use_all_gpus'] = False
        args['parallel_criterion'] = False
        args['freeze_bn'] = False

        if checkpoint_path == 'extensions/rawr/data/lrnet/experiments/10-12,13:20;config_name:lrnet/checkpoints/model_50.pth':
            args['predictor_config']['return_features'] = False
            args['processor_config']['num_maps_input'] = 3
            args['processor_config']['filter_size'] = 3
            args['processor_config']['model'] += '_'
        
        if 'processor_config' in args:    
            args['processor_config']['freeze_bn'] = False
        
        if 'predictor_config' in args:
            args['predictor_config']['freeze_bn'] = False

        if args_to_update is not None:
            args.update(args_to_update)

        m_model = load_module(args['extension'], 'models', args['model'])

        print(args)
        model = m_model.get_net(munchify(args), None)

        return model, args




class ModelAndLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelAndLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, target, *data):
        output = self.model(*data)
        loss = self.loss(output, target)

        return loss, output














import numpy as np
# if state_dict['feature_extractor.0.weight'].shape[1] != args.num_input_channels:
#     print('Surgery ==============')
#     t = torch.zeros( (state_dict['feature_extractor.0.weight'].shape[0], args.num_input_channels, state_dict['feature_extractor.0.weight'].shape[2], state_dict['feature_extractor.0.weight'].shape[3]), dtype=torch.float)

#     for i in range(int(args.num_input_channels / 3)):
#         t[:, i * 3: (i + 1) * 3] = state_dict['feature_extractor.0.weight'] / (int(args.num_input_channels / 3))

#     if args.num_input_channels % 3 > 0: 
#         t[:, - (args.num_input_channels % 3) : ] = state_dict['feature_extractor.0.weight'][:, -(args.num_input_channels % 3) :]


#     state_dict['feature_extractor.0.weight'] = t
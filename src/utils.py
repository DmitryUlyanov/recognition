import importlib
import sys
import random
import os.path
import cv2
import time
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import imgaug as ia

def setup(args):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    args.cuda = True
    torch.backends.cudnn.benchmark = True

    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)

    print("Random Seed: ", args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    ia.seed(args.random_seed)


def get_optimizer(args, model):
    # Parse parameters 
    optimizer_args = {}
    for entry in args.optimizer_args.split("^"):
      k, v = entry.split('=')
      optimizer_args[k] = eval(v)

    s  = sum([np.prod(list(p.size())) for p in model.parameters()])
    print ('Number of params: %d' % s)

    # Optimizer
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), **optimizer_args)

    return optimizer    

def get_args_and_modules(parser):
    '''
        Gathers args from modules and config
    '''
    args_, _ = parser.parse_known_args()

    # Update main defaults
    m_config = load_config(args_.extension, args_.config_name)
    if m_config is not None:
        m_config.update_defaults(parser)

    # Parse with new defaults
    args_, _ = parser.parse_known_args()

    # Add model args
    m_model = load_module(args_.extension, 'models', args_.model)
    m_model.get_args(parser)

    # Add dataloader args
    m_dataloader = load_module(args_.extension, 'dataloaders', args_.dataloader) 
    m_dataloader.get_args(parser)

    # Add runner args
    m_runner = load_module(args_.extension, 'runners', args_.runner)
    m_runner.get_args(parser)

    # Again import defaults
    if m_config is not None:
        m_config.update_defaults(parser)

    # Finally parse everything
    args, default_args = parser.parse_args(), parser.parse_args([])

    return args, default_args, dict(runner=m_runner, dataloader=m_dataloader, model=m_model)

def load_config(extension, config_name):
    if extension != "" and os.path.exists(f'extensions/{extension}/{config_name}.py'):
        print (f'Using config extensions/{extension}/{config_name}.py.')
        return importlib.import_module(f'extensions.{extension}.{config_name}')
    else:
        print ('Config not found.')
        return None 

def load_module(extension, module_type, module_name):
    '''
        module_type : models | dataloaders 
    '''
    if extension == '':
        m = importlib.import_module(f'{module_type}.{module_name}')
    else:
        if os.path.exists(f'extensions/{extension}/{module_type}/{module_name}.py'):
            m = importlib.import_module(f'extensions.{extension}.{module_type}.{module_name}')
        else:
            print(f'Extension module {extension}/{module_type}/{module_name} not found.')
            
            if os.path.exists(f'{module_type}/{module_name}.py'):
                m = importlib.import_module(f'{module_type}.{module_name}')
            else:       
                print(f"Default module {module_name} not found.")
                assert False
            

    return m



def fn(self):
    return self.cpu().data.numpy()

torch.autograd.Variable.to_numpy = fn


# def load_model(args):
#     '''
#     Loads generator model.
#     '''
#     m = importlib.import_module('models.' + args.model)
#     model, criterion = m.get_net(args)

    
#     return model, criterion

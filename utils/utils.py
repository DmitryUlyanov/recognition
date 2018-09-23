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
import yaml
import re 

def setup(args):
    torch.set_num_threads(0)
    cv2.setNumThreads(0)

    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)

    print("Random Seed: ", args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)

    ia.seed(args.random_seed)


def get_optimizer(args, model):
    # Parse parameters
    optimizer_args = {}
    for entry in args.optimizer_args.split("^"):
        k, v = entry.split('=')
        optimizer_args[k] = eval(v)

    s = sum([np.prod(list(p.size())) for p in model.parameters()])
    print ('Number of params: %d' % s)

    # Optimizer
    optimizer = torch.optim.__dict__[args.optimizer](
        model.parameters(), **optimizer_args)

    return optimizer

# from collections import namedtuple
# def to_namedtuple(dictionary):
#     return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def get_args_and_modules(parser, phase='train'):
    '''
        Gathers args from modules and config
    '''
    args_, _ = parser.parse_known_args()

    # Update main defaults
    update_defaults_fn = load_saved_args(args_.experiment_dir) if phase == 'test' else load_config(args_.extension, args_.config_name, args_)
    if update_defaults_fn is not None:
        update_defaults_fn(parser)

    # Parse with new defaults
    args_, _ = parser.parse_known_args()

    # Add model args
    m_model = load_module(args_.extension, 'models', args_.model)
    m_model.get_args(parser)

    # Add dataloader args
    m_dataloader = load_module(
        args_.extension, 'dataloaders', args_.dataloader)
    m_dataloader.get_args(parser)

    # Add runner args
    m_runner = load_module(args_.extension, 'runners', args_.runner)
    m_runner.get_args(parser)

    update_defaults_fn = load_saved_args(args_.experiment_dir) if phase == 'test' else load_config(args_.extension, args_.config_name, args_)
    if update_defaults_fn is not None:
        update_defaults_fn(parser)


    # Finally parse everything
    args, default_args = parser.parse_args(), parser.parse_args([])

    args_dict = var(args)
    if phase == 'test':
        saved_args = load_yaml(f'{args.experiment_dir}/args.yaml')
        for k in .keys():
            if k not in args_dict:
                print(k)
                args_dict[k] = saved_args[k]

        args = argparse.Namespace(**args_dict)

    return args, default_args, dict(runner=m_runner, dataloader=m_dataloader, model=m_model)


def load_saved_args(experiment_dir):
    yaml_config=f'{experiment_dir}/args.yaml'

    print (f'Using config {yaml_config}')
    with open(yaml_config, 'r') as stream:
        config = yaml.load(stream)

    def update_defaults_fn(parser):

        for k in config.keys():
            if isinstance(config[k], str):
                config[k] = re.sub('\$\{(.*?)\}', lambda x: str(vars(args)[x.groups()[0]]), config[k], flags=re.DOTALL)
                
        parser.set_defaults(**config)
        return parser

    return update_defaults_fn



def load_config(extension, config_name, args):
    if extension != "":
        py_config = f'extensions/{extension}/{config_name}.py'
        yaml_config = f'extensions/{extension}/{config_name}.yml'

        if os.path.exists(py_config):
            print (f'Using config {py_config}')
            return importlib.import_module(f'extensions.{extension}.{config_name}').update_defaults

        elif os.path.exists(yaml_config):
            print (f'Using config {yaml_config}')
            with open(yaml_config, 'r') as stream:
                config = yaml.load(stream)

            def update_defaults_fn(parser):

                for k in config.keys():
                    if isinstance(config[k], str):
                        config[k] = re.sub('\$\{(.*?)\}', lambda x: str(vars(args)[x.groups()[0]]), config[k], flags=re.DOTALL)
                        
                parser.set_defaults(**config)
                return parser

            return update_defaults_fn
        else:
            assert False
    else:
        print ('Config not found.')
        return None


def load_config(extension, config_name, args):
    if extension != "":
        py_config = f'extensions/{extension}/{config_name}.py'
        yml_config = f'extensions/{extension}/{config_name}.yml'

        if os.path.exists(py_config):
            print (f'Using config {py_config}')
            return importlib.import_module(f'extensions.{extension}.{config_name}').update_defaults

        elif os.path.exists(yml_config):
            print (f'Using config {yml_config}')
            with open(yml_config, 'r') as stream:
                config = yaml.load(stream)

            def update_defaults_fn(parser):

                for k in config.keys():
                    if isinstance(config[k], str):
                        config[k] = re.sub('\$\{(.*?)\}', lambda x: str(vars(args)[x.groups()[0]]), config[k], flags=re.DOTALL)
                        
                parser.set_defaults(**config)
                return parser

            return update_defaults_fn
        else:
            assert False
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

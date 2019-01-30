import importlib
import sys
import random
import os
import cv2
import time
import torch
import numpy as np
import imgaug as ia
import yamlenv
# import ruamel.yaml as yamlenv
import re 
import argparse
from huepy import red, green
from .io_utils import load_yaml, save_yaml
# import utils.optimizers


FILE_PATH = os.path.abspath(os.path.dirname(__file__))
RECOGNITION_PATH = os.path.abspath(FILE_PATH + '/..')

def setup(args):
    torch.set_num_threads(0)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)

    print("Random Seed: ", args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)

    ia.seed(args.random_seed)

def parse_dict(s):
    d = {}

    if len(s) > 0:
        for entry in s.split("^"):
            k, v = entry.split('=')
            d[k] = eval(v)

    return d




def get_args_and_modules(parser, phase='train', saved_args=None):
    '''
        Gathers args from modules and config
    '''

    if phase == 'test':
        saved_args = load_saved_args(args_.experiment_dir, args_) if saved_args is None else saved_args

    args_, _ = parser.parse_known_args()

    # Update main defaults
    update_defaults_fn = get_update_defaults_fn(saved_args) if phase == 'test' else load_config(args_.extension, args_.config_name, args_)
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

    update_defaults_fn = get_update_defaults_fn(saved_args) if phase == 'test' else load_config(args_.extension, args_.config_name, args_)
    if update_defaults_fn is not None:
        update_defaults_fn(parser)


    # Finally parse everything
    args, default_args = parser.parse_args(), parser.parse_args([])

    args_dict = vars(args)
    # if phase == 'test':
    #     saved_args = load_yaml(f'{args.experiment_dir}/args.yaml')
    #     print(saved_args)
    #     for k in saved_args.keys():
    #         if k not in args_dict:
    #             print(k)
    #             args_dict[k] = saved_args[k]

    #     args = argparse.Namespace(**args_dict)

    return args, default_args, dict(runner=m_runner, dataloader=m_dataloader, model=m_model)


def load_saved_args(experiment_dir, args):
    yaml_config=f'{experiment_dir}/args.yaml'

    print ((f'Using config {green(yaml_config)}'))

    with open(config, 'r') as stream:
        config = yamlenv.load(stream)

    return config 
    


def get_update_defaults_fn(config):

    if isinstance(config, str):
        with open(config, 'r') as stream:
            config = yamlenv.load(stream)
       
    def update_defaults_fn(parser):
        parser.set_defaults(**config)
        return parser

    return update_defaults_fn


def load_config(extension, config_name, args):
    
    if extension == '':
        assert False, red(f'Extension is not specified.')


    config_extension = f'extensions/{extension}/{config_name}.yaml'
    config_lib       = f'configs/{config_name}.yaml'
    
    for config in [config_extension, config_lib]:
        if os.path.exists(config):
            print ((f'Using config {green(config)}'))
            return get_update_defaults_fn(config)
        else:
            print ((f'Did not find config {green(config)}'))

    assert False, red(f'Config not found!')




        
# def load_model_class(extension, module_name):
#     '''
#         module_type : models | dataloaders 
#     '''
#     cdir = os.getcwd()
#     os.chdir(RECOGNITION_PATH)

#     from models.model import load_model 

#     # Try to search in defined ones
#     model = load_model( )

#     # Search in extension
#     if model is None:
#         m = load_module(extension, module_name)
#         model = m.module_name()

#     os.chdir(cdir)

#     return model


def load_module_(extension, module_type, module_name, raise_error=True):
    '''
        module_type : models | dataloaders 
    '''


    if extension == '':
        m = importlib.import_module(f'{module_type}.{module_name}')
    else:
        if os.path.exists(f'extensions/{extension}/{module_type}/{module_name}.py'):
            m = importlib.import_module(f'extensions.{extension}.{module_type}.{module_name}')
            print(f"Extension module {green(module_name)} loaded.")
        else:
            # print(f'Extension module {extension}/{module_type}/{module_name} not found.')

            if os.path.exists(f'{module_type}/{module_name}.py'):
                m = importlib.import_module(f'{module_type}.{module_name}')
                print((f"Default module {green(module_name)} loaded."))
            else:
                if raise_error:
                    assert False, red(f"Default or extension module {module_name} not found.")
                else:
                    return None

    return m




def load_module(extension, module_type, module_name):
    '''
        module_type : models | dataloaders 
    '''
    cdir = os.getcwd()
    # os.chdir(RECOGNITION_PATH)

    if module_type == 'models':
        from models.model import Model
        return Model(module_name, extension) 

    if module_type == 'dataloaders':
        from dataloaders.dataloader import Dataloader 
        return Dataloader(module_name, extension) 

    m = load_module_(extension, module_type, module_name)
    
    # os.chdir(cdir)

    return m


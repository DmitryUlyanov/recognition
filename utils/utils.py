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
import utils.optimizers


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


def get_optimizer(args, model):
    
    # Parse parameters
    optimizer_args = parse_dict(args.optimizer_args)
   

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    s = sum([np.prod(list(p.size())) for p in params_to_optimize])
    print (f' - Number of params: {s}')

    optimizer = utils.optimizers.get_optimizer(args.optimizer)(params_to_optimize, **optimizer_args)
    
    return optimizer

def get_scheduler(args, optimizer):

    # Parse parameters
    scheduler_args = parse_dict(args.scheduler_args)
    
    scheduler = utils.optimizers.get_scheduler(args.scheduler)(optimizer, **scheduler_args)
    
    return scheduler


def get_args_and_modules(parser, phase='train'):
    '''
        Gathers args from modules and config
    '''
    args_, _ = parser.parse_known_args()

    # Update main defaults
    update_defaults_fn = load_saved_args(args_.experiment_dir, args_) if phase == 'test' else load_config(args_.extension, args_.config_name, args_)
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

    update_defaults_fn = load_saved_args(args_.experiment_dir, args_) if phase == 'test' else load_config(args_.extension, args_.config_name, args_)
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

    return get_update_defaults_fn(yaml_config, args)


def get_update_defaults_fn(yaml_config, args):
    with open(yaml_config, 'r') as stream:
        config = yamlenv.load(stream)

    # def fill_templates(config_dict):
    #     for k in config_dict.keys():
    #         if isinstance(config_dict[k], str):
    #             config_dict[k] = re.sub('\$\{(.*?)\}', lambda x: str(vars(args).get(x.groups()[0], '${' + x.groups()[0] + '}')), config_dict[k], flags=re.DOTALL)
    #         elif isinstance(config_dict[k], dict):
    #             fill_templates(config_dict[k])

    def update_defaults_fn(parser):
        # fill_templates(config)

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
            return get_update_defaults_fn(config, args)
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


def load_module_(extension, module_type, module_name):
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
                assert False, red(f"Default or extension module {module_name} not found.")

    return m




def load_module(extension, module_type, module_name):
    '''
        module_type : models | dataloaders 
    '''
    cdir = os.getcwd()
    os.chdir(RECOGNITION_PATH)

    if module_type == 'models':
        # print('2222222222222222')
        from models.model import Model
        return Model(module_name, extension) 

    m = load_module_(extension, module_type, module_name)
    
    os.chdir(cdir)

    return m


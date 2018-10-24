import importlib
import sys
import random
import os
import cv2
import time
import torch
import numpy as np
import imgaug as ia
import yaml
import re 
import argparse
from huepy import red, green
from .io_utils import load_yaml, save_yaml

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


def get_optimizer(args, model):
    # Parse parameters
    optimizer_args = {}
    for entry in args.optimizer_args.split("^"):
        k, v = entry.split('=')
        optimizer_args[k] = eval(v)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    s = sum([np.prod(list(p.size())) for p in params_to_optimize])
    print ('Number of params: %d' % s)

    # Optimizer
    optimizer = torch.optim.__dict__[args.optimizer](
        params_to_optimize, **optimizer_args)

    return optimizer


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
    if phase == 'test':
        saved_args = load_yaml(f'{args.experiment_dir}/args.yaml')
        print(saved_args)
        for k in saved_args.keys():
            if k not in args_dict:
                print(k)
                args_dict[k] = saved_args[k]

        args = argparse.Namespace(**args_dict)

    return args, default_args, dict(runner=m_runner, dataloader=m_dataloader, model=m_model)


def load_saved_args(experiment_dir, args):
    yaml_config=f'{experiment_dir}/args.yaml'

    print ((f'Using config {green(yaml_config)}'))

    return get_update_defaults_fn(yaml_config, args)


def get_update_defaults_fn(yaml_config, args):
    with open(yaml_config, 'r') as stream:
        config = yaml.load(stream)

    def fill_templates(config_dict):
        for k in config_dict.keys():
            if isinstance(config_dict[k], str):
                config_dict[k] = re.sub('\$\{(.*?)\}', lambda x: str(vars(args).get(x.groups()[0], '${' + x.groups()[0] + '}')), config_dict[k], flags=re.DOTALL)
            elif isinstance(config_dict[k], dict):
                fill_templates(config_dict[k])

    def update_defaults_fn(parser):
        fill_templates(config)

        parser.set_defaults(**config)
        return parser

    return update_defaults_fn


def load_config(extension, config_name, args):
    
    if extension == '':
        assert False, red(f'Extension is not specified.')

    yaml_config = f'extensions/{extension}/{config_name}.yaml'

    if os.path.exists(yaml_config):
        print ((f'Using config {green(yaml_config)}'))
        return get_update_defaults_fn(yaml_config, args)
    elif os.path.exists(f'configs/{config_name}.yaml'):
        print ((f'Using config {green(yaml_config)}'))
        return get_update_defaults_fn(yaml_config, args)
    else:
        assert False, red(f'Config {config_name} not found.')

        


def load_module(extension, module_type, module_name):
    '''
        module_type : models | dataloaders 
    '''
    cdir = os.getcwd()
    os.chdir(RECOGNITION_PATH)

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

    os.chdir(cdir)

    return m

class ActionNoYes(argparse.Action):
    def __init__(self, 
                option_strings,
                dest,
                nargs=0,
                const=None,
                default=None,
                type=None,
                choices=None,
                required=False,
                help="",
                metavar=None):

        assert len(option_strings) == 1
        assert option_strings[0][:2] == '--'
        
        name= option_strings[0][2:]
        help += f'Use "--{name}" for True, "--no-{name}" for False'
        super(ActionNoYes, self).__init__(['--' + name, '--no-' + name], 
                                          dest=dest,
                                          nargs=nargs,
                                          const=const,
                                          default=default,
                                          type=type,
                                          choices=choices, 
                                          required=required, 
                                          help=help,
                                          metavar=metavar)
        
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


class SplitStr(argparse.Action):
    
    def split(self, x):
        if x == '':
            return []
        else:
            return [self.elem_type(y) for y in x.split(self.delimiter)]

    def __init__(self, 
                option_strings,
                dest,
                nargs=None,
                const=None,
                default=None,
                type=None,
                choices=None,
                required=False,
                help="",
                metavar=None,
                delimiter=',',
                elem_type=str):

        self.delimiter = delimiter
        self.elem_type = elem_type
        
        default = self.split(default)
        super(SplitStr, self).__init__(option_strings, 
                                          dest=dest,
                                          nargs=nargs,
                                          const=const,
                                          default=default,
                                          type=type,
                                          choices=choices, 
                                          required=required, 
                                          help=help,
                                          metavar=metavar)
        
    def __call__(self, parser, namespace, values, option_string=None):
        print(values)
        setattr(namespace, self.dest, self.split(values))


class MyArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(MyArgumentParser, self).__init__(**kwargs)
        self.register('action', 'store_bool', ActionNoYes)
        self.register('action', 'split_str',  SplitStr)

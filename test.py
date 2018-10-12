#!/usr/bin/env python

import argparse
import importlib
import os
import torch
# import numpy as np
from utils.utils import setup, get_args_and_modules, MyArgumentParser
from exp_logger import setup_logging, print_experiment_info
from utils import save_drivers
from utils import io_utils
from models import criterions
from tensorboardX import SummaryWriter

# Define main args
parser = MyArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--experiment_dir', type=str, default="", help='manual seed')
# parser.add('--checkpoint', type=str, default="", help='manual seed')

# parser.add('--model', type=str, default="", help='')
# parser.add('--dataloader', type=str, default="", help='')
# parser.add('--runner', type=str, default="", help='')
# parser.add('--extension', type=str, default="", help='manual seed')

# parser.add('--random_seed', type=int, default=123, help='manual seed')

# parser.add('--experiments_dir', type=str, default="experiments", help='')
parser.add('--part', type=str, default='val', help='test|val|train')


# parser.add('--config_name', type=str, default="config")

parser.add('--need_softmax', default=False, action='store_bool')

parser.add('--save_driver', default=None, type=str)
parser.add('--dump_path', default=None, type=str)


# Gather args across modules
args, default_args, m = get_args_and_modules(parser, phase='test')

# Setup everything else
setup(args)

# Print what is changed
print_experiment_info(args, default_args, args.experiment_dir)

# Load splits and preprocess target
model_native_transform = m['model'].get_native_transform()
dataloader = m['dataloader'].get_dataloader(args, model_native_transform, args.part)

# Load model 
model = m['model'].get_net(args, dataloader)
model.eval()

criterion = criterions.get_loss(args.criterion).to(args.device)

if args.save_driver is not None:
    save_driver = getattr(save_drivers, args.save_driver)
else:
    save_driver = None


writer = SummaryWriter(log_dir = args.experiment_dir, filename_suffix='test')

m['runner'].run_epoch.writer = writer

torch.set_grad_enabled(False)
loss = m['runner'].run_epoch(
                            dataloader, 
                            model,
                            criterion,
                            None,
                            epoch=0, 
                            args=args,part='test'           )
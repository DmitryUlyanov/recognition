#!/usr/bin/env python


import argparse
import importlib
import os
import torch
# import numpy as np
from utils.argparse_utils import MyArgumentParser
from utils.utils import setup, get_args_and_modules

from exp_logger import setup_logging, print_experiment_info
from utils import save_drivers
from utils import io_utils
from models import criterions
import colored_traceback.auto
import utils.savers as savers
from pathlib import Path
# Define main args
parser = MyArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--experiment_dir', type=Path, help='manual seed')



# parser.add('--model', type=str, default="", help='')
# parser.add('--dataloader', type=str, default="", help='')
# parser.add('--runner', type=str, default="", help='')
# parser.add('--extension', type=str, default="", help='manual seed')

parser.add('--random_seed', type=int, default=123, help='manual seed')

# parser.add('--experiments_dir', type=str, default="experiments", help='')
parser.add('--part', type=str, default='val', help='test|val|train')

parser.add('--set_eval_mode', action='store_bool', default=True)
# parser.add('--config_name', type=str, default="config")

# parser.add('--need_softmax', default=False, action='store_bool')

parser.add('--save_driver', default=None, type=str)
parser.add('--dump_path', default=None, type=str)

# Saver 
parser.add('--saver',       type=str, default='DummySaver', help='Just any type of comment')
parser.add('--saver_args',  type=str, default='')



# Gather args across modules
args, default_args, m = get_args_and_modules(parser, phase='test')


args.checkpoint_load_only_extractor = False
args.checkpoint_strict_load_state = True

# Setup everything else
setup(args)

# Print what is changed
print_experiment_info(args, default_args, args.experiment_dir)

# Load splits and preprocess target
model_native_transform = m['model'].get_native_transform()
dataloader = m['dataloader'].get_dataloader(args, model_native_transform, args.part)


if args.part == 'test':
    criterion = criterions.get_criterion('DummyCriterion', args).to(args.device)
else:
    criterion = criterions.get_criterion(args.criterion, args).to(args.device)


# Load model 
model = m['model'].get_net(args, dataloader, criterion)

if args.set_eval_mode:
    model.eval()
else:
    model.train()
    

class DummyWriter(object):
    """docstring for DummyWriter"""
    def __init__(self, *args):
        super(DummyWriter, self).__init__()

    def add_scalar(self, *args):
        pass
     
    def add_image(self, *args):
        pass   


saver = savers.get_saver(args.saver, args.saver_args)

torch.set_grad_enabled(False)
loss = m['runner'].run_epoch(
                            dataloader=dataloader, 
                            model=model,
                            criterion=criterion,
                            optimizer=None,
                            epoch=0, 
                            writer=DummyWriter(),
                            saver=saver,
                            args=args,
                            phase='test')
# 



# criterion = criterions.get_loss(args.criterion).to(args.device)

# if args.save_driver is not None:
#     save_driver = getattr(save_drivers, args.save_driver)
# else:
#     save_driver = None

# writer = SummaryWriter(log_dir = args.experiment_dir, filename_suffix='test')

# for i in range(13):
#     args.checkpoint = f'model_{i}.pth'

#     # Load model 
#     model = m['model'].get_net(args, dataloader)
#     model.eval()



    

#     m['runner'].run_epoch.writer = writer

#     torch.set_grad_enabled(False)
#     loss = m['runner'].run_epoch(
#                                 dataloader, 
#                                 model,
#                                 criterion,
#                                 None,
#                                 epoch=0, 
#                                 args=args,part='test'           )
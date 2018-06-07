import argparse
import importlib
import os
import torch
import pickle
import numpy as np
# from src.training_testing import run_epoch
from src.utils import setup, get_optimizer, get_args_and_modules
from exp_logger import setup_logging

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--model', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')
parser.add('--extension', type=str, default="", help='manual seed')

parser.add('--random_seed', type=int, default=123, help='manual seed')

# parser.add('--experiments_dir', type=str, default="experiments", help='')
parser.add('--part', type=str, default='test', help='test|val|train')

parser.add('--preds_save_path', default="", type=str)
parser.add('--config_name', type=str, default="config")

parser.add('--no_need_softmax', default=False, action='store_true')
parser.add('--save_driver', default='pickle', type=str)

# Gather args across modules
args, default_args, m = get_args_and_modules(parser)

# Get experiment dir
# postfix = get_postfix(vars(args), vars(default_args), args_to_ignore)
# exp_dir = os.path.join(args.experiments_dir, postfix)

# Setup everything else
setup(args)

# Load splits and preprocess target
dataloader = m['dataloader'].get_dataloader(args, args.part)

# Load model 
model, criterion = m['model'].get_net(args)
model.eval()

torch.set_grad_enabled(False)
loss, preds = m['runner'].run_epoch_test(dataloader, model, criterion, epoch=0, args=args, need_softmax=not args.no_need_softmax, need_preds=True)

if args.preds_save_path != "":
    print(f'Saving predictions to {args.preds_save_path}')

    if args.save_driver == 'npz':
        np.savez_compressed(args.preds_save_path, preds=preds)
    else:
        with open(args.preds_save_path, 'wb') as f:
            pickle.dump(preds, f, -1)
        
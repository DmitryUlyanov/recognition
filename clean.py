import argparse
import importlib
import os
import torch
from LSUV import LSUVinit
from src.utils import setup, get_optimizer, get_args_and_modules
from exp_logger import setup_logging

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--exp_dir', type=str, default="", help='')
parser.add('--no-dry-run', action='store_true')

# Gather args across modules
args = parser.parse_args()

import glob 

for exp_path in glob.glob(f'{args.exp_dir}/*'):

    checkpoints = glob.glob(f'{exp_path}/checkpoints/*')

    if len(checkpoints) != 0:
        print(exp_path)

        if args.no_dry_run:
            shutil.rmtree(exp_path)
            
# Setup logging and save dir
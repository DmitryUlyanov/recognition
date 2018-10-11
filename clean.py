import argparse
import importlib
import os
import torch
import shutil
from huepy import red
# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--exp_dir', type=str, default="", help='')
parser.add('--no-dry-run', action='store_true')

args = parser.parse_args()

import glob 

for exp_path in glob.glob(f'{args.exp_dir}/*'):

    checkpoints = glob.glob(f'{exp_path}/checkpoints/*')

    if len(checkpoints) == 0:
        print('Deleting ', red(exp_path))

        if args.no_dry_run:
            shutil.rmtree(exp_path)

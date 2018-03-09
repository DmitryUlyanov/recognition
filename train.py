import argparse
import importlib
import os
import torch

from src.training_testing import run_epoch
from src.utils import setup, get_optimizer
from exp_logger import setup_logging

# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--model', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')

parser.add('--print_frequency', type=int, default=1, help='manual seed')
parser.add('--manual_seed', type=int, default=123, help='manual seed')
parser.add('--experiments_dir', type=str, default="experiments", help='')
parser.add('--comment', type=str, default='', help='Just any type of comment')

parser.add('--optimizer', type=str, default='SGD', help='Just any type of comment')
parser.add('--optimizer_args', default="lr=1e-1", type=str, help='separated with ";" list of args i.e. "lr=1e-3;betas=(0.5,0.9)"')

parser.add('--num_epochs', type=int, default=100, help='manual seed')

parser.add('--mode', type=str, default="regression",
           help='classification|regression|regression_masked')

parser.add('--no-logging', default=False, action="store_true")

args_, _ = parser.parse_known_args()

# Add model args
m_model = importlib.import_module('models.' + args_.model)
m_model.get_args(parser)

# Add dataloader args
m_dataloader = importlib.import_module('dataloaders.' + args_.dataloader)
m_dataloader.get_args(parser)

args, default_args = parser.parse_args(), parser.parse_args([])

# Setup logging and save dir
args.save_dir = 'data' if args.no_logging else setup_logging(args, default_args, [])
os.makedirs(f'{args.save_dir}/checkpoints', exist_ok=True)

# Setup everything else
setup(args)

# Load splits and preprocess target
dataloader_train, dataloader_val = m_dataloader.get_dataloaders(args)

# Load model 
model, criterion = m_model.get_net(args)

# Load optimizer 
optimizer = get_optimizer(args, model)

for epoch in range(0, args.num_epochs):
    # if ep == 100:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr']*0.1

    run_epoch(dataloader_train, model, criterion, optimizer, epoch, args.mode, 'train')
    run_epoch(dataloader_val,   model, criterion, optimizer, epoch, args.mode, 'test')
    
    torch.save(model.state_dict(), f'{args.save_dir}/checkpoints/model_{epoch}.pth')


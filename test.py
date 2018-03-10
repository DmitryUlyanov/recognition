import argparse
import importlib
import os
import torch

# from src.training_testing import run_epoch
from src.utils import setup, get_optimizer
from exp_logger import setup_logging

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--model', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')

parser.add('--print_frequency', type=int, default=1, help='manual seed')

parser.add('--manual_seed', type=int, default=123, help='manual seed')

parser.add('--part', type=str, default='test', help='test|val|train')

parser.add('--mode', type=str, default="regression",
           help='classification|regression|regression_masked')

parser.add('--preds_save_path', default="", type=str)

args_, _ = parser.parse_known_args()

# Add model args
m_model = importlib.import_module('models.' + args_.model)
m_model.get_args(parser)

# Add dataloader args
m_dataloader = importlib.import_module('dataloaders.' + args_.dataloader)
m_dataloader.get_args(parser)

args = parser.parse_args()

# Setup everything else
setup(args)

# Load splits and preprocess target
dataloader = m_dataloader.get_dataloader(args, "test")

# Load model 
model, criterion = m_model.get_net(args)

m_runner = importlib.import_module('runners.' + args_.mode)

    
model.eval()
val, preds = m_runner.run_epoch_test (dataloader, model, criterion, epoch=0, need_softmax=True, need_preds=True)

if args.preds_save_path != "":
    print(f'Saving predictions to {args.preds_save_path}')
    np.save(args.preds_save_path, preds)
# if (epoch != 0) and (epoch % args.save_frequency == 0):
#     torch.save(model.state_dict(), f'{args.save_dir}/checkpoints/model_{epoch}.pth')


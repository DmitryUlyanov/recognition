import argparse
import importlib
import os
import torch
import pickle

# from src.training_testing import run_epoch
from src.utils import setup, get_optimizer, load_module
from exp_logger import setup_logging

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--model', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')
parser.add('--extension', type=str, default="", help='manual seed')

parser.add('--manual_seed', type=int, default=123, help='manual seed')


parser.add('--part', type=str, default='test', help='test|val|train')

parser.add('--preds_save_path', default="", type=str)

args_, _ = parser.parse_known_args()

m = importlib.import_module(f'extensions.{args_.extension}.config')
m.update_defaults(parser)

args_, _ = parser.parse_known_args()

# Add model args
m_model = load_module(args_.extension, 'models', args_.model)
m_model.get_args(parser)

# Add dataloader args
m_dataloader = load_module(args_.extension, 'dataloaders', args_.dataloader) 
m_dataloader.get_args(parser)

# Add runner args
m_runner = load_module(args_.extension, 'runners', args_.runner)
m_runner.get_args(parser)

args = parser.parse_args()

# Setup everything else
setup(args)

# Load splits and preprocess target
dataloader = m_dataloader.get_dataloader(args, args.part)

# Load model 
model, criterion = m_model.get_net(args)

model.eval()
loss, preds = m_runner.run_epoch_test(dataloader, model, criterion, epoch=0, args=args, need_softmax=True, need_preds=True)

if args.preds_save_path != "":
    print(f'Saving predictions to {args.preds_save_path}')
    with open(args.preds_save_path, 'wb') as f:
        pickle.dump(preds, f, -1)
        
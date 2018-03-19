import argparse
import importlib
import os
import torch

from src.utils import setup, get_optimizer, load_module
from exp_logger import setup_logging

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define main args
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--model', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')

parser.add('--save_frequency',  type=int, default=5, help='manual seed')

parser.add('--manual_seed', type=int, default=123, help='manual seed')
parser.add('--experiments_dir', type=str, default="experiments", help='')
parser.add('--comment', type=str, default='', help='Just any type of comment')

parser.add('--optimizer', type=str, default='SGD', help='Just any type of comment')
parser.add('--optimizer_args', default="lr=1e-1", type=str, help='separated with ";" list of args i.e. "lr=1e-3;betas=(0.5,0.9)"')

parser.add('--extension', type=str, default="", help='manual seed')
parser.add('--num_epochs', type=int, default=100, help='manual seed')

parser.add('--mode', type=str, default="regression",
           help='classification|regression|regression_masked')

parser.add('--no-logging', default=False, action="store_true")

# parser.add('--args-to-ignore', default='')

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

args, default_args = parser.parse_args(), parser.parse_args([])

# Setup logging and save dir
args.save_dir = 'data' if args.no_logging else setup_logging(args, default_args, [])
os.makedirs(f'{args.save_dir}/checkpoints', exist_ok=True)

# Setup everything else
setup(args)

# Load splits and preprocess target
dataloader_train = m_dataloader.get_dataloader(args, 'train')
dataloader_val   = m_dataloader.get_dataloader(args, 'val')

# Load model 
model, criterion = m_model.get_net(args)

# Load optimizer and scheduler
optimizer = get_optimizer(args, model)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

for epoch in range(0, args.num_epochs):
    model.train()
    m_runner.run_epoch_train(dataloader_train, model, criterion, optimizer, epoch, args)
    
    model.eval()
    val_loss = m_runner.run_epoch_test (dataloader_val,   model, criterion, epoch, args)
    
    scheduler.step(val_loss)

    if (epoch != 0) and (epoch % args.save_frequency == 0):
        torch.save(model.state_dict(), f'{args.save_dir}/checkpoints/model_{epoch}.pth', pickle_protocol=-1)


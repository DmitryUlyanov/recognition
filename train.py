import argparse
import importlib
import os
import torch
from utils.utils import MyArgumentParser
from tensorboardX import SummaryWriter
from models.model import save_model
from huepy import yellow 

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("forkserver")

import models.criterions as criterions
import json
from utils.utils import setup, get_optimizer, get_args_and_modules, get_scheduler
from utils.io_utils import save_yaml
from exp_logger import setup_logging
import signal
import time
import sys
# from torch.optim.lr_scheduler import ReduceLROnPlateau



def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if input("\nReally quit? (y/n)> ").lower().startswith('y'):
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, exit_gracefully)



original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)




# Define main args
parser = MyArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--extension',  type=str, default="")
parser.add('--config_name', type=str, default="")

parser.add('--model', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')
parser.add('--criterion', type=str, default="", help='')

parser.add('--save_frequency',  type=int, default=1, help='')

parser.add('--random_seed',     type=int, default=123, help='')
parser.add('--experiments_dir', type=str, default="experiments", help='')
parser.add('--comment', type=str, default='', help='Just any type of comment')

parser.add('--optimizer', type=str, default='SGD', help='Just any type of comment')
parser.add('--optimizer_args', default="lr=3e-3^momentum=0.9", type=str, help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')


parser.add('--scheduler', type=str, default='ReduceLROnPlateau', help='Just any type of comment')
parser.add('--scheduler_args', default="factor=0.5^min_lr=1e-6^verbose=True^patience=0", type=str, help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')


parser.add('--criterion_args', default="", type=str, help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')


parser.add('--num_epochs', type=int, default=200)

# parser.add('--patience',         type=int, default=5)
# parser.add('--lr_reduce_factor', type=float, default=0.3)

parser.add('--logging', default=True, action="store_bool")
parser.add('--args-to-ignore', type=str, default="checkpoint,splits_dir,experiments_dir,extension")

parser.add('--set_eval_mode_in_train', action='store_bool', default=False)
parser.add('--set_eval_mode_in_test', action='store_bool', default=True)

parser.add('--need_pretrain', action='store_bool', default=False)

parser.add('--device', type=str, default='cuda')


parser.add('--save_driver', default=None, type=str)
parser.add('--dump_path', default=None, type=str)

parser.add('--set_eval_mode_epoch', default=-1, type=int)


# Gather args across modules
args, default_args, m = get_args_and_modules(parser)

# NUM CLASSES
with open(args.data_path + '/groups_to_run.json') as f:
    groups_to_run = json.load(f)

group_len = groups_to_run[args.group_num][1]
args.num_classes = str(group_len + 1)
print(args.num_classes)
################

# Setup logging and creates save dir
if args.logging:
    args.experiment_dir, writer = setup_logging(args, 
                                                default_args, 
                                                args.args_to_ignore.split(','), 
                                                exp_name_use_date=True)
else:
    args.experiment_dir = '/tmp/recognition'
    writer = SummaryWriter(log_dir = args.experiment_dir, filename_suffix='_train')
# Dump args
save_yaml(vars(args), f'{args.experiment_dir}/args.yaml')

# Setup everything else
setup(args)

# Load splits and preprocess target
model_native_transform = m['model'].get_native_transform()
dataloader_train       = m['dataloader'].get_dataloader(args, model_native_transform, 'train')
dataloader_val         = m['dataloader'].get_dataloader(args, model_native_transform, 'val')


# Load criterion
if args.criterion != "": 
    criterion = criterions.get_criterion(args.criterion, args).to(args.device)

# Load model 
model = m['model'].get_net(args, dataloader_train, criterion)

# Load optimizer and scheduler
optimizer = get_optimizer(args, model)

scheduler = get_scheduler(args, optimizer)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.lr_reduce_factor, verbose=True, min_lr=1e-6)

# Dump args
save_yaml(vars(args), f'{args.experiment_dir}/args_modified.yaml')

args.get_dataloader = m['dataloader'].get_dataloader
m['runner'].run_epoch.writer = writer




def set_param_grad(model, value, set_eval_mode=True):
    for param in model.parameters():
        param.requires_grad = value
    
    if set_eval_mode:
        model.eval()


if args.need_pretrain:
    
    # Load optimizer and scheduler
    # args.fix_feature_extractor = True
    set_param_grad(model.module.feature_extractor, False, False)
    optimizer = get_optimizer(args, model)

    scheduler = get_scheduler(args, optimizer)


    # Stage1
    torch.set_grad_enabled(True)
    m['runner'].run_epoch(dataloader_train, model, criterion, optimizer, -1, args, part='train')
    
    # Validate
    if args.set_eval_mode_in_test:
        model.eval()
    else:
        model.train()

    torch.set_grad_enabled(False)
    val_loss = m['runner'].run_epoch(dataloader_val, model, criterion, None, -1, args, part='val')

    save_model(model, -1, args)



    # Stage 2 

    set_param_grad(model, True, set_eval_mode=False)
    # Load optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)






for epoch in range(0, args.num_epochs):
    if args.set_eval_mode_in_train or (args.set_eval_mode_epoch >= 0 and epoch>=args.set_eval_mode_epoch):
        print(yellow(f' - Setting eval mode!'))
        model.eval()
    else:
        model.train()

    # Train

    torch.set_grad_enabled(True)
    m['runner'].run_epoch(dataloader_train, model, criterion, optimizer, epoch, args, part='train')
    
    # Validate
    if args.set_eval_mode_in_test:
        model.eval()
    else:
        model.train()

    torch.set_grad_enabled(False)
    val_loss = m['runner'].run_epoch(dataloader_val, model, criterion, None, epoch, args, part='val')
    
    scheduler.step(val_loss)

    # Save

    if epoch % args.save_frequency == 0:
        save_model(model, epoch, args, optimizer)

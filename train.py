


    # torch.multiprocessing.set_start_method("forkserver")
# if __name__ == '__main__':

from exp_logger import setup_logging
from huepy import yellow 
from models.model import save_model
from munch import munchify
from pathlib import Path
from utils.argparse_utils import MyArgumentParser
from utils.io_utils import save_yaml
from utils.utils import setup, get_args_and_modules

from contrib.optimizers import get_optimizer
from contrib.schedulers import get_scheduler
from contrib.savers     import get_saver

from contrib.criterions.criterions import get_criterion

import argparse
import colored_traceback.auto
import importlib
import json

import os
import signal
import sys
import time
import torch
import torch.multiprocessing


torch.multiprocessing.set_sharing_strategy('file_system')

# def exit_gracefully(signum, frame):
#     # restore the original signal handler as otherwise evil things will happen
#     # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
#     signal.signal(signal.SIGINT, original_sigint)

#     try:
#         if input("\nReally quit? (y/n)> ").lower().startswith('y'):
#             sys.exit(1)

#     except KeyboardInterrupt:
#         print("Ok ok, quitting")
#         sys.exit(1)

#     # restore the exit gracefully handler here    
#     signal.signal(signal.SIGINT, exit_gracefully)



# original_sigint = signal.getsignal(signal.SIGINT)
# signal.signal(signal.SIGINT, exit_gracefully)




# Define main args
parser = MyArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--extension',  type=str, default="")
parser.add('--config_name', type=str, default="")

parser.add('--model',      type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner',     type=str, default="", help='')

parser.add('--experiments_dir', type=Path, default="experiments", help='')

# Criterion 
parser.add('--criterion',      type=str, default="", help='')
parser.add('--criterion_args', type=str, default="",  help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')

# Optimizer 
parser.add('--optimizer',      type=str, default='SGD', help='Just any type of comment')
parser.add('--optimizer_args', type=str, default="lr=3e-3^momentum=0.9")

# Scheduler 
parser.add('--scheduler', type=str, default='ReduceLROnPlateau', help='Just any type of comment')
parser.add('--scheduler_args', default="factor=0.5^min_lr=1e-6^verbose=True^patience=3", type=str, help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')

# Training loop
parser.add('--stage', type=str, default=None)
parser.add('--num_epochs',      type=int, default=200)
parser.add('--set_eval_mode_in_train', action='store_bool', default=False)
parser.add('--set_eval_mode_in_test',  action='store_bool', default=True)
parser.add('--set_eval_mode_epoch', default=-1, type=int)
parser.add('--save_frequency',  type=int, default=1, help='')

# Logging
parser.add('--logging', default=True, action="store_bool")
parser.add('--args-to-ignore', type=str, default="checkpoint,splits_dir,experiments_dir,extension")
parser.add('--comment',         type=str, default='', help='Just any type of comment')

# Misc
parser.add('--random_seed',     type=int, default=123, help='')
parser.add('--device', type=str, default='cuda')




# Gather args across modules
args, default_args, m = get_args_and_modules(parser)


# Setup logging and creates save dir
if args.logging:
    args.experiment_dir, writer = setup_logging(args, 
                                                default_args, 
                                                args.args_to_ignore.split(','), 
                                                exp_name_use_date=True)

    args.experiment_dir = Path(args.experiment_dir)


# Dump args
save_yaml(vars(args), f'{args.experiment_dir}/args.yaml')

# Setup everything else
setup(args)


# Load dataloaders
model_native_transform = m['model'].get_native_transform()
dataloader_train       = m['dataloader'].get_dataloader(args, model_native_transform, part='train', phase='train')
dataloader_val         = m['dataloader'].get_dataloader(args, model_native_transform, part='val',   phase='train')


# Load criterion
criterion = get_criterion(args.criterion, args).to(args.device)


# Load model 
model = m['model'].get_net(args, dataloader_train, criterion)

# Load saver
saver = get_saver('DummySaver')

# Dump args (if modified)
save_yaml(vars(args), f'{args.experiment_dir}/args_modified.yaml')




def set_param_grad(model, value, set_eval_mode=True):
    for param in model.parameters():
        param.requires_grad = value
    
    if set_eval_mode:
        model.eval()


if 'stages' not in vars(args): 
    args.stages = {'main': {}}


if args.stage is not None:

    if args.stage != 'none':
        args.stages = {args.stage: args.stages[args.stage]}
    else:
        args.stages = {'main': {}}


for stage_num, (stage_name, stage_args_) in enumerate(args.stages.items()):

    print (yellow(f' - Starting stage "{stage_name}"!'))

    stage_args = munchify({**vars(args), **stage_args_ })

    if hasattr(model, 'module'): 
    #stage_args.fix_feature_extractor:
        set_param_grad(model.module.feature_extractor, value=not stage_args.fix_feature_extractor, set_eval_mode=False)
        # set_param_grad(model.module.feature_extractor[-1], value=True, set_eval_mode=False)
    

    optimizer = get_optimizer(stage_args, model)
    scheduler = get_scheduler(stage_args, optimizer)


    # Go
    for epoch in range(0, stage_args.num_epochs):

        if stage_args.set_eval_mode_in_train or (stage_args.set_eval_mode_epoch >= 0 and epoch>=stage_args.set_eval_mode_epoch):
            print(yellow(f' - Setting eval mode!'))
            model.eval()
        else:
            model.train()


        # ===================
        #       Train
        # ===================
        torch.set_grad_enabled(True)
        m['runner'].run_epoch(dataloader_train, model, criterion, optimizer, epoch, stage_args, phase='train', writer=writer, saver=saver)
        


        # ===================
        #       Validate
        # ===================
        if stage_args.set_eval_mode_in_test:
            model.eval()
        else:
            model.train()

        torch.set_grad_enabled(False)
        val_loss = m['runner'].run_epoch(dataloader_val, model, criterion, None, epoch, stage_args, phase='val', writer=writer, saver=saver)
        
        scheduler.step(val_loss)


        # Save
        if epoch % stage_args.save_frequency == 0:
            save_model(model, epoch, stage_args, optimizer, stage_num)
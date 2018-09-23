import importlib
import sys
import random
import os.path
import time
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import tqdm

from dataloaders.colorspace import rgb2lab
from torch.autograd import Variable
from torch.nn.functional import softmax
from runners.common import AverageMeter, accuracy
from PIL import Image
import typed_print as tp

print = tp.init(palette='dark', # or 'dark' 
                str_mode=tp.HIGHLIGHT_NUMBERS, 
                highlight_word_list=['Epoch'])

def get_args(parser):
  parser.add('--print_frequency', type=int, default=50)
  parser.add('--niter_in_epoch', type=int, default=0)

  return parser

def run_epoch_train(dataloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
            
    outputs = []

    end = time.time()
    

    for it, (names, x_, y_) in enumerate(dataloader):
        
        batch_size = x_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)

        x_var, y_var = x_.cuda(async=True), y_.cuda(async=True)

        output = model(x_var)       
        loss = criterion(output, y_var)

        losses.update(loss.item(), x_var.shape[0])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'Epoch: [{epoch}][{it}/{len(dataloader) if args.niter_in_epoch <= 0 else args.niter_in_epoch}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
        
        if args.niter_in_epoch > 0 and it % args.niter_in_epoch == 0 and it > 0:
          break          
    
    print(f' * \n'
          f' * Epoch {epoch} Training:\t'
          f'Loss {losses.avg:.4f}\t'
          f' *\t\n')

        
def run_epoch_test(dataloader, model, criterion, epoch, args, need_softmax=False, save_driver=None):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    
    avg_loss   = AverageMeter()
     
    
    outputs, all_names = [], []

    end = time.time()
    # print('erere', len(dataloader))
    for it, (names, x, y) in enumerate(dataloader):
        
        # Measure data loading time
        data_time.update(time.time() - end)

        x_var, y_var = x.cuda(async=True), y.cuda(async=True)
        
        output = model(x_var)

        # print(output.shape, x_var)
        loss = criterion(output, y_var)
        # print(y_var.max(),y_var.min())
        if need_softmax:
            output = [softmax(o) for o in output]
            
        avg_loss.update(loss.item(), x.shape[0])
       
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'Epoch: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {avg_loss.val:.4f} ({avg_loss.avg:.4f})')

        all_names.append(names)

        if need_preds:
            save_driver(x_var, y_var, output, names, args.preds_save_path)
            

    print(f' * \n'
          f' * Epoch {epoch} Testing:\t'
          f'Loss {avg_loss.avg:.4f}\t'
          f' *\t\n')

    return loss.item()


import importlib
import sys
import random
import os.path
# from dataloader import *
import time
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

# import torch.distributions
# from torch.distributions import beta
from torch.nn.functional import softmax
from .common import AverageMeter, accuracy

import typed_print as tp

print = tp.init(palette='dark', # or 'dark' 
                str_mode=tp.HIGHLIGHT_NUMBERS, 
                highlight_word_list=['Epoch'])

def get_args(parser):
  parser.add('--use_mixup',  default=False, action='store_true')
  parser.add('--mixup_alpha', type=float, default=0.1)

  parser.add('--print_frequency', type=int, default=50)
  
  return parser

def run_epoch_train(dataloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
            
    outputs = []

    end = time.time()
    
    for it, (names, x_, y_) in enumerate(dataloader):
        
        batch_size = x_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)

        if args.use_mixup:
            x1, x2 = x_[:batch_size // 2].cuda(async=True), x_[batch_size // 2:].cuda(async=True)
            y1, y2 = y_[:batch_size // 2].cuda(async=True), y_[batch_size // 2:].cuda(async=True)
          
            lam = torch.from_numpy(np.random.beta(args.mixup_alpha + 1, args.mixup_alpha, [batch_size // 2, 1, 1, 1]).astype(np.float32)).cuda(async=True)

            x_var = x1 * lam + x2 * (1 - lam)
            y_var = y1
        else: 
            x_var = x_.cuda(async=True)
            y_var = y_.cuda(async=True)


        output = model(x_var)

        # y_var = y_var.type(torch.cuda.LongTensor)
        
        loss = criterion(output, y_var)

        losses.update(loss.item())#, x_var.shape[0])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'Epoch: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
      
    
    print(f' * \n'
          f' * Epoch {epoch} Training:\t'
          f'Loss {losses.avg:.4f}\t'
          f' *\t\n')

        
def run_epoch_test(dataloader, model, criterion, epoch, args, need_softmax=False, need_preds=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    avg_loss = AverageMeter()
    top1 = AverageMeter()
     
    outputs, all_names = [], []

    end = time.time()
    for it, (names, x, y) in enumerate(dataloader):
        
        # Measure data loading time
        data_time.update(time.time() - end)

        x_var, y_var = x.cuda(async=True), y.cuda(async=True)
            
        output = model(x_var)

        y_var = y_var.type(torch.cuda.LongTensor)
        loss = loss = criterion(output, y_var)

        if need_softmax:
            output = [softmax(o) for o in output]
            
        if need_preds:
            outputs.append([o.cpu().numpy() for o in output])

        avg_loss.update(loss.item())#, x.shape[0])

        top1_ = [accuracy(output[i].data, y_var[:, i].data.contiguous(), topk=(1,)) for i in range(len(output))]
        top1.update(np.mean([o[0].item() for o in top1_]), x.shape[0])
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'Epoch: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {avg_loss.val:.4f} ({avg_loss.avg:.4f})\t'\
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t')
                  
        
        all_names.append(names)

    print(f' * \n'
          f' * Epoch {epoch} Testing:\t'
          f'Loss {avg_loss.avg:.4f}\t'
          f'Prec@1 {top1.avg:.3f}\t'
          f' *\t\n')

    if need_preds:
        # Get list of names
        all_names = sum(all_names, [])

        # Rearrange the data a little
        p = []
        for i, pr in enumerate(outputs):
            num_obj = pr[0].shape[0] # batch size
            
            for j in range(num_obj):
                p.append([pr_class[j] for pr_class in pr])

        '''
        Get dict  of type
            
            name: [ preds_for_class_1, preds_for_class_2, ... ] 
        '''
        d = {k: v for k, v in zip(all_names, p)}
        

        return loss.item(), d

    return loss.item()


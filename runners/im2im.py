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
import tqdm

from dataloaders.colorspace import rgb2lab
# import torch.distributions
# from torch.distributions import beta
from torch.autograd import Variable
from torch.nn.functional import softmax
from runners.common import AverageMeter, accuracy
from PIL import Image
import typed_print as tp

print = tp.init(palette='dark', # or 'dark' 
                str_mode=tp.HIGHLIGHT_NUMBERS, 
                highlight_word_list=['Epoch'])

def get_args(parser):
  # parser.add('--use_mixup',  default=False, action='store_true')
  # parser.add('--mixup_alpha', type=float, default=0.1)

  parser.add('--print_frequency', type=int, default=50)

  parser.add('--niter_in_epoch', type=int, default=0)

  parser.add('--loss_colorspace', type=str, default='rgb')
  
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

        x_var, y_var = x_.cuda(async=True), y_.cuda(async=True)

        output = model(x_var)
        
        # losses_ = []
        
        if args.loss_colorspace == 'lab':
            output, y_var = rgb2lab(output), rgb2lab(y_var) 
        
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

        
def run_epoch_test(dataloader, model, criterion, epoch, args, need_softmax=False, need_preds=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    avg_loss = AverageMeter()
    top1 = AverageMeter()
     
    
    outputs, all_names = [], []

    end = time.time()
    # print('erere', len(dataloader))
    for it, (names, x, y) in enumerate(dataloader):
        
        # Measure data loading time
        data_time.update(time.time() - end)

        x_var, y_var = x.cuda(async=True), y.cuda(async=True)
        
        output = model(x_var)

        if args.loss_colorspace == 'lab':
            output, y_var = rgb2lab(output), rgb2lab(y_var) 

        # print(output.shape, x_var)
        loss = criterion(output, y_var)
        # print(y_var.max(),y_var.min())
        if need_softmax:
            output = [softmax(o) for o in output]
            
        # if need_preds:
        #     outputs.append([o.cpu().numpy() for o in output])

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
            for x, y, o, name in zip(x_var, y_var, output, names):
                print(22)
                x_name = f'{args.preds_save_path}/{os.path.basename(name)}_x.png'
                y_name = f'{args.preds_save_path}/{os.path.basename(name)}_gt.png'
                o_name = f'{args.preds_save_path}/{os.path.basename(name)}_aut.png'

                Image.fromarray((torch.clamp(x[:3].detach().cpu(), 0, 1).numpy().transpose(1, 2, 0)*255).astype(np.uint8)).save(x_name, quality=100, optimize=True, progressive=True)
                Image.fromarray((torch.clamp(y[:3].detach().cpu(), 0, 1).numpy().transpose(1, 2, 0)*255).astype(np.uint8)).save(y_name, quality=100, optimize=True, progressive=True)
                Image.fromarray((torch.clamp(o[:3].detach().cpu(), 0, 1).numpy().transpose(1, 2, 0)*255).astype(np.uint8)).save(o_name, quality=100, optimize=True, progressive=True)
		
            break

    print(f' * \n'
          f' * Epoch {epoch} Testing:\t'
          f'Loss {avg_loss.avg:.4f}\t'
          f'Prec@1 {top1.avg:.3f}\t'
          #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
          f' *\t\n')

    # Get list of names
    # all_names = sum(all_names, [])

    # # Rearrange the data a little
    # p = []
    # for i, pr in enumerate(outputs):
    #     num_obj = pr[0].shape[0] # batch size
        
    #     for j in range(num_obj):
    #         p.append([pr_class[j] for pr_class in pr])

    '''
    Get dict  of type
        
        name: [ preds_for_class_1, preds_for_class_2, ... ] 
    '''
    # d = {k: v for k, v in zip(all_names, p)}
    
    # from numpy import unravel_index

    # # s2 = np.concatenate(outputs, axis=0)


    # # Return max
      
    # # if False:
    # preds = []

    # for b in tqdm.tqdm(outputs):
    #   # s2 = s2_[0]
    #   for o in b:
    #     p = []
    #     for j in range(o.shape[0]):
    #         p.append(unravel_index(o[j].argmax(), o[j].shape))
            
    #     preds.append(p)

    # print(len(preds))
    # return loss.item(), preds
    
    # else:
    #   for b in tqdm.tqdm(outputs):
    #     # s2 = s2_[0]
    #     for o in b:
    #       p = []
    #       for j in range(o.shape[0]):

    #           cv2.imwrite
    #           p.append(unravel_index(o[j].argmax(), o[j].shape))
              
    #       preds.append(p)

    #   print(len(preds))
    #   return loss.data[0], preds

    return loss.item()


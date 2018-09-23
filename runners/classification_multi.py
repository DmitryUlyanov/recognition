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
from huepy import red, lightblue, orange
# print = tp.init(palette='dark', # or 'dark' 
#                 str_mode=tp.HIGHLIGHT_NUMBERS) 
                # highlight_word_list=['Training'])

def get_args(parser):
  parser.add('--use_mixup',  default=False, action='store_true')
  parser.add('--mixup_alpha', type=float, default=0.1)

  parser.add('--print_frequency', type=int, default=50)
  
  return parser

def check_data(data): #dataloader):
   (names, x_, y_)  = data #iter(dataloader_train).next()[1].to(args.device)

   assert isinstance(names, list) and isinstance(x_, list) and isinstance(y_, list), red("expecting each output to be a list of tensors")
   assert len(names) == len(x_)
   assert len(x_) == 1 or len(x_) == len(y_)

def run_epoch_train(dataloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter = AverageMeter()
            
    outputs = []

    end = time.time()
    
    for it, (names, x_, y_) in enumerate(dataloader):
        optimizer.zero_grad()

        check_data((names, x_, y_))
        
        # None when there are no samples from some dataset, keep them anyway
        y_ = [y.to(args.device) if y is not None else y for y in y_]
        x_ = [x.to(args.device) if x is not None else x for x in x_]
        # batch_size = x_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)


        total_loss = 0

        # torch.cat([x.to(args.device) if x is not None else x for x in x_])
        # for i, x in enumerate(x_):
        #   if x is None:
        #     continue

        #   output = model(x.to(args.device)) # list 


        #   if len(x_) == 1: 
        #     # Same data, different tasks
        #     loss = criterion(output, y_) # y_var here is a list of tensors
            
        #   else:
        #     # different data different tasks
        #     loss = criterion([output[i]], [y_[i]]) # y_var here is a list of one tensor
            
        #   total_loss += loss.item()

        #   (loss / len(y_)).backward()


        for i, x in enumerate(x_):
          if x is None:
            continue

          output = model(x.to(args.device)) # list 


          if len(x_) == 1: 
            # Same data, different tasks
            loss = criterion(output, y_) # y_var here is a list of tensors
            
          else:
            # different data different tasks
            loss = criterion([output[i]], [y_[i]]) # y_var here is a list of one tensor
            
          total_loss += loss.item()

          (loss / len(y_)).backward()

        optimizer.step()
        

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loss_meter.update(total_loss / len(x_))#, x_var.shape[0])

        if it % args.print_frequency == 0:
            print(f'{lightblue("Training")}: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t')
      
    
    print(f' * \n'
          f' * Epoch {epoch} {red("Training")}:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f' *\t\n')

        
def run_epoch_test(dataloader, model, criterion, epoch, args, need_softmax=False, save_driver=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    avg_loss = AverageMeter()
    top1 = AverageMeter()
     
    outputs, all_names = [], []

    end = time.time()
    for it, (names, x_, y_) in enumerate(dataloader):
        check_data((names, x_, y_))

        # Measure data loading time
        data_time.update(time.time() - end)

        # x_var, y_var = x.cuda(async=True), y.cuda(async=True)
            
        # output = model(x_var)

        # y_var = y_var.type(torch.cuda.LongTensor)
        # loss = loss = criterion(output, y_var)

        y_ = [y.to(args.device) if y is not None else y for y in y_]
        x_ = [x.to(args.device) if x is not None else x for x in x_]
        # batch_size = x_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)


        total_loss = 0
        for i, x in enumerate(x_):
          if x is None:
            continue

          output = model(x.to(args.device)) # list 


          if len(x_) == 1: 
            # Same data, different tasks
            loss = criterion(output, y_) # y_var here is a list of tensors
            
          else:
            # different data different tasks
            loss = criterion([output[i]], [y_[i]]) # y_var here is a list of one tensor
            

          if save_driver is not None:
          # for h in range(len(names)):
            for _name, _x, _y, _pred in zip(names[i], x_[i], y_[i], output):

              save_driver({
                'target': y.detach().cpu().numpy(),
                'pred': _pred.detach().cpu().numpy(),
                }, path=f'{args.save_preds_path}/{_name}.npz')
        

        # if need_softmax:
        #     output = [softmax(o) for o in output]
            
            # outputs.append([o.cpu().numpy() for o in output])

        avg_loss.update(loss.item())#, x.shape[0])

        # top1_ = [accuracy(output[i].data, y_var[:, i].data.contiguous(), topk=(1,)) for i in range(len(output))]
        # top1.update(np.mean([o[0].item() for o in top1_]), x.shape[0])
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'{orange("Testing")}: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {avg_loss.val:.4f} ({avg_loss.avg:.4f})\t')
                  # f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t')
                  
        
        all_names.append(names)

    print(f' * \n'
          f' * Epoch {epoch} {red("Testing")}:\t'
          f'Loss {avg_loss.avg:.4f}\t'
          f'Prec@1 {top1.avg:.3f}\t'
          f' *\t\n')

    # if need_preds:
    #     # Get list of names
    #     all_names = sum(all_names, [])

    #     # Rearrange the data a little
    #     p = []
    #     for i, pr in enumerate(outputs):
    #         num_obj = pr[0].shape[0] # batch size
            
    #         for j in range(num_obj):
    #             p.append([pr_class[j] for pr_class in pr])

    #     '''
    #     Get dict  of type
            
    #         name: [ preds_for_class_1, preds_for_class_2, ... ] 
    #     '''
    #     d = {k: v for k, v in zip(all_names, p)}
        

    #     return loss.item(), d

    return loss.item()


import sys
import time
import torch
import tqdm
from runners.common import print_stat, get_grid, tensor_to_device_recursive, Meter
from huepy import lightblue, cyan, red
import numpy as np


def get_args(parser):
  parser.add('--log_frequency_loss',   type=int,   default=50)
  parser.add('--log_frequency_images', type=int, default=1000)


  parser.add('--niter_in_epoch', type=int, default=0)

  parser.add('--gradient_accumulation', type=int, default=1)


  return parser



def run_epoch(dataloader, model, criterion, optimizer, epoch, args, part, writer, saver = None):
    
    meter = Meter()

    if part=='train':
        optimizer.zero_grad()

    end = time.time()
    for it, data in enumerate(dataloader):

        # Measure data loading time
        meter.update('Data time', time.time() - end)

        names, x_, y_ = data['names'], data['input'], data['target']
        x, y = tensor_to_device_recursive(x_), tensor_to_device_recursive(y_)

        # Forward
        if args.merge_model_and_loss:
            loss, output = model(y, x)
            loss = loss.mean()
        else:
            output = model(x)      
            loss = criterion(output, y)


        # Backward
        if part == 'train':
            (loss / args.gradient_accumulation).backward()
                        
            if it % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()


        saver.maybe_save(iteration=it, output=output, names=names)
        

        # ----------------------------
        #            Logging 
        # ----------------------------
        
        meter.update('Loss', loss.item())

        if part == 'train':

            for metric in meter.data.keys():
                writer.add_scalar(f'Metrics/{part}/{metric}', meter.get_last(metric),   writer.last_it)
            
            writer.add_scalar(f'LR', optimizer.param_groups[0]['lr'],   writer.last_it)

            writer.last_it += 1



        # Print
        if it % args.log_frequency_loss == 0:

            s = f'{lightblue(part.capitalize())}: [{epoch}][{it}/{len(dataloader)}]\t'

            for metric in meter.data.keys():
                s += f'{print_stat(metric, meter.get_last(metric), meter.get_avg(metric), 4)}    '

            print(s)

                    
              
        # Measure elapsed time
        meter.update('Batch time', time.time() - end)
        end = time.time()


        if part == 'train' and args.niter_in_epoch > 0 and it % args.niter_in_epoch == 0 and it > 0:
            break   

    saver.stop()


    # Printing    
    s = f' * \n * Epoch {epoch} {red(part.capitalize())}:\t'
    for metric in meter.data.keys():
        s += f'{metric} {meter.get_avg(metric):.4f}    '

    print(s + ' *\t\n')


    if part != 'train':
        s = f'{lightblue(part.capitalize())}: [{epoch}][{it}/{len(dataloader)}]\t'

        for metric in meter.data.keys():
            writer.add_scalar(f'Metrics/{part}/{metric}', meter.get_avg(metric),   epoch)


    return meter.get_avg('Loss')
import sys
import time
import torch
import tqdm
from runners.common import print_stat, get_grid, tensor_to_device_recursive, Meter, accuracy
from huepy import lightblue, cyan, red
import numpy as np


def get_args(parser):
    parser.add('--log_frequency_loss',   type=int,   default=50)
    parser.add('--log_frequency_images', type=int, default=1000)

    parser.add('--gradient_accumulation', type=int, default=1)

    parser.add('--metrics', type=str, default='')

    parser.add('--detailed_metrics', action='store_bool', default=True)

    return parser



def run_epoch(dataloader, model, criterion, optimizer, epoch, args, phase, writer, saver = None):
    
    meter = Meter()

    f = False

    if phase=='train':
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
            losses_dict = criterion(output, y)
            loss = sum(losses_dict.values()) / len(losses_dict)

        # Backward
        if phase == 'train':
            (loss / args.gradient_accumulation).backward()
                        
            if it % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()


        saver.maybe_save(iteration=it, output=output, names=names, labels=y)
        
        # ----------------------------
        #            Logging 
        # ----------------------------
        
        meter.update('Loss', loss.item())

        if args.detailed_metrics:
            for loss_name, loss_ in losses_dict.items():
                meter.update(f'Loss_{loss_name}', loss_.item())    

        
        

        if 'accuracy' in args.metrics:
            accs = accuracy(output, y)
            
            meter.update(f'Top1_avg', np.mean(accs))

            if args.detailed_metrics:
                for i, acc in enumerate(accs):
                    meter.update(f'Top1_{i}',  acc)

            


        if phase == 'train':

            for metric in meter.data.keys():
                writer.add_scalar(f'Metrics/{phase}/{metric}', meter.get_last(metric),   writer.last_it)
            
            writer.add_scalar(f'LR', optimizer.param_groups[0]['lr'],   writer.last_it)

            writer.last_it += 1



        # Print
        if it % args.log_frequency_loss == 0:

            s = f'{lightblue(phase.capitalize())}: [{epoch}][{it}/{len(dataloader)}]\t'

            for metric in meter.data.keys():
                s += f'{print_stat(metric, meter.get_last(metric), meter.get_avg(metric), 4)}    '

            print(s)

                    
              
        # Measure elapsed time
        meter.update('Batch time', time.time() - end)
        end = time.time()


    saver.stop()


    # Printing    
    s = f' * \n * Epoch {epoch} {red(phase.capitalize())}:\t'
    for metric in meter.data.keys():
        s += f'{metric} {meter.get_avg(metric):.4f}    '

    print(s + ' *\t\n')


    if phase != 'train':
        s = f'{lightblue(phase.capitalize())}: [{epoch}][{it}/{len(dataloader)}]\t'

        for metric in meter.data.keys():
            writer.add_scalar(f'Metrics/{phase}/{metric}', meter.get_avg(metric),   epoch)


    return meter.get_avg('Loss')
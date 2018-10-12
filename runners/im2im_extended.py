import os.path
import time
import torch
import numpy as np
import tqdm
from runners.common import AverageMeter, accuracy, print_stat
from huepy import red, lightblue, orange, cyan

def get_args(parser):
  parser.add('--print_frequency', type=int, default=50)
  parser.add('--niter_in_epoch',  type=int, default=0)
  
  return parser


def check_data(data): 
   (names, y_, x_, params_)  = data 

   assert len(names) == len(x_) == len(y_) == len(params_)


def run_epoch_train(dataloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter = AverageMeter()
            
    outputs = []

    end = time.time()
    

    for it, (names, y_, x_, params_) in enumerate(dataloader):
        optimizer.zero_grad()
        check_data((names, y_, x_, params_))

        # Measure data loading time
        data_time.update(time.time() - end)

        x, y, params = x_.cuda(non_blocking=True), y_.cuda(non_blocking=True), params_.cuda(non_blocking=True)

        # Forward
        output = model(x, params)
        loss   = criterion(output, y)
        
        # Backward and step
        loss.backward()
        optimizer.step()

        # Update loss stats
        loss_meter.update(loss.item(), x_.shape[0])

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'{lightblue("Train")}: [{cyan(epoch)}][{cyan(it)}/{cyan(len(dataloader) if args.niter_in_epoch <= 0 else args.niter_in_epoch)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t')
                    
        if args.niter_in_epoch > 0 and it % args.niter_in_epoch == 0 and it > 0:
          break          
    
    print(f' * \n'
          f' * Epoch {epoch} {red("Training")}:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f' *\t\n')


def run_epoch(dataloader, model, criterion, optimizer, epoch, args, part='train'):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter = AverageMeter()
                
    writer = run_epoch.writer
    outputs = []

    end = time.time()
    

    for it, (names, y_, *x_) in enumerate(dataloader):
        
        batch_size = y_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)

        x = [x.to(args.device, non_blocking=True) for x in x_]
        y = y_.cuda(non_blocking=True)

        output = model(*x)      

        loss = criterion(output, y)
    
        if part=='train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # Logging 
        loss_meter.update(loss.item(), y.shape[0])
        writer.add_scalar(f'Loss', loss_meter.val,  last_it[part])
        last_it[part] += 1
        if it % args.log_frequency_loss == 0:
            print(f'{lightblue(part.capitalize())}: [{cyan(epoch)}][{cyan(it)}/{cyan(len(dataloader) if args.niter_in_epoch <= 0 else args.niter_in_epoch)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t')
                    

        if it % args.log_frequency_images == 0:
            writer.add_image(f'{part}_{epoch}', get_grid(x_[0], y_, output),  it)
              
        if args.niter_in_epoch > 0 and it % args.niter_in_epoch == 0 and it > 0:
            break 


        if save_driver in args and args.save_driver is not None:
            for num, (_name, _x, _y, _pred) in enumerate(zip(names, x_[0], y_, output)):

              save_driver({
                'input':  _x.detach().cpu().numpy(),
                'target': _y.detach().cpu().numpy(),
                'pred':   _pred.detach().cpu().numpy(),
                'name':   _name,
                }, path=f'{args.dump_path}/{hash(_name)}.png', args=args)         
    

    print(f' * \n'
          f' * Epoch {epoch}, {part.capitalize()} stat:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f' *\t\n')

    return loss_meter.avg
import os.path
import time
import torch
import numpy as np
import tqdm
from runners.common import AverageMeter, accuracy, print_stat
from huepy import red, lightblue, orange, cyan


def get_args(parser):
  parser.add('--use_mixup',  default=False, action='store_true')
  parser.add('--mixup_alpha', type=float, default=0.1)

  parser.add('--print_frequency', type=int, default=50)
  
  return parser

# def check_data(data): #dataloader):
#    (names, x_, y_)  = data #iter(dataloader_train).next()[1].to(args.device)

#    assert isinstance(names, list) and isinstance(x_, list) and isinstance(y_, list), red("expecting each output to be a list of tensors")
#    assert len(names) == len(x_)
#    assert len(x_) == 1 or len(x_) == len(y_)

def run_epoch_train(dataloader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    outputs = []

    end = time.time()
    
    for it, (names, x_, y_) in enumerate(dataloader):
        optimizer.zero_grad()

        # check_data((names, x_, y_))
        
        
        y_ = y_.to(args.device, non_blocking=True)
        x_ = x_.to(args.device, non_blocking=True)
        
        # Measure data loading time
        data_time.update(time.time() - end)

        output = model(x_)[0]
        loss = criterion(output, y_[:, 0]) 
        
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loss_meter.update(loss.item())
        acc_meter.update(accuracy(output, y_)[0].item())


        if it % args.print_frequency == 0:
            print(f'{lightblue("Train")}: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t',
                  f'{print_stat("Acc",  acc_meter.val,  acc_meter.avg, 4)}\t')
      
    
    print(f' * \n'
          f' * Epoch {epoch} {red("Training")}:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f'Acc  {acc_meter.avg:.4f}\t'
          f' *\t\n')

        
def run_epoch_test(dataloader, model, criterion, epoch, args, need_softmax=False, save_driver=None):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
     
    outputs, all_names = [], []

    end = time.time()
    for it, (names, x_, y_) in enumerate(dataloader):
        # check_data((names, x_, y_))

        # Measure data loading time
        data_time.update(time.time() - end)

        y_ = y_.to(args.device, non_blocking=True)
        x_ = x_.to(args.device, non_blocking=True)
        
        # Measure data loading time
        data_time.update(time.time() - end)

        output = model(x_)[0]
        loss = criterion(output, y_[:, 0]) 

        if save_driver is not None:
            for num, (_name, _x, _y, _pred) in enumerate(zip(names, x_, y_, output)):
                save_driver({
                  'target': _y.detach().cpu().numpy(),
                  'pred':   _pred.detach().cpu().numpy(),
                  'name':   _name,
                  }, path=f'{args.dump_path}/{hash(_name)}.npz')
        

        # if need_softmax:
        #     output = [softmax(o) for o in output]
            
            # outputs.append([o.cpu().numpy() for o in output])

        loss_meter.update(loss.item())
        acc_meter.update(accuracy(output, y_)[0].item())

        # top1_ = [accuracy(output[i].data, y_var[:, i].data.contiguous(), topk=(1,)) for i in range(len(output))]
        # top1.update(np.mean([o[0].item() for o in top1_]), x.shape[0])
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_frequency == 0:
            print(f'{orange("Test")}: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t',
                  f'{print_stat("Acc",  acc_meter.val,  acc_meter.avg, 4)}\t')
                  
        
        # all_names.append(names)

    print(f' * \n'
          f' * Epoch {epoch} {red("Testing")}:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f'Acc  {acc_meter.avg:.3f}\t'
          f' *\t\n')


    return loss_meter.avg


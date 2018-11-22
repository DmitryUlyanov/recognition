import os.path
import time
import torch
import numpy as np
import tqdm
from runners.common import AverageMeter, accuracy, print_stat
from huepy import red, lightblue, orange, cyan
from utils.task_queue import TaskQueue


def get_args(parser):
    parser.add('--use_mixup',  default=False, action='store_true')
    parser.add('--mixup_alpha', type=float, default=0.1)

    parser.add('--print_frequency', type=int, default=50)
    parser.add('--niter_in_epoch', type=int, default=0)
    
    parser.add('--gradient_accumulation', type=int, default=1)

    return parser

def check_data(data): #dataloader):
   (names, x_, y_)  = data #iter(dataloader_train).next()[1].to(args.device)

   assert isinstance(names, list) and isinstance(x_, torch.cuda.FloatTensor) and isinstance(y_, list), red("expecting each output to be a list of tensors")
   assert len(names) == len(y_)
   assert len(x_) == 1 or len(x_) == len(y_)

data = dict(last_it=0)


import torch
from torch import nn


class MAPk(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk

    def forward(self, input, target):
        _, predictions = input.topk(self.topk, dim=1, sorted=True)

        apk_v = torch.eq(target, predictions[:, 0]).float()
        for k in range(1, self.topk):
            apk_v += torch.eq(target, predictions[:, k]).float() / (k + 1)

        return apk_v.mean()



def run_epoch(dataloader, model, criterion, optimizer, epoch, args, part='train'):
    batch_time  = AverageMeter()
    data_time   = AverageMeter()
    loss_meter  = AverageMeter()
    acc_meter   = AverageMeter()
    topk_meter  = AverageMeter()
    
    writer = run_epoch.writer
    outputs = []


    saver = Saver(args, npz_per_batch, tq_maxsize = 5)

    end = time.time()
    
    if part=='train':
        optimizer.zero_grad()
        dataloader = args.get_dataloader(args, None, 'train')

    for it, (names, x_, *y_) in enumerate(dataloader):
        
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # check_data((names, x_, y_))
        
        y = [y.to(args.device, non_blocking=True) for y in y_]
        x = x_.to(args.device, non_blocking=True)
        
        

        if args.merge_model_and_loss:
            loss, output = model(y, x)
            loss = loss.mean()
        else:
            output = model(x)      
            loss = criterion(output, y)


        if part=='train':
            (loss / args.gradient_accumulation).backward()
                        
            if it % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loss_meter.update(loss.item())

        # For multitarget tasks we want to get acc per target
        acc_per_target = accuracy(output, y)
        acc_meter.update(np.mean([x for x in acc_per_target if x != -1]))

        topk = MAPk(3)(output[0], y[0])

        saver.maybe_save(iteration=it, x=x_, y=y_, output=output, names=names)

        # Logging 
        loss_meter.update(loss.item(), x.shape[0])

        topk_meter.update(topk, x.shape[0])

        if part == 'train':
            writer.add_scalar(f'Metrics/{part}/loss', loss.item(),   data['last_it'])
            writer.add_scalar(f'Metrics/{part}/acc', acc_meter.val,  data['last_it'])
            writer.add_scalar(f'Metrics/{part}/mapk', topk_meter.val,  data['last_it'])

            # writer.add_scalars(f'metrics/{part}/accs_per_target', {str(k): acc_per_target[k] for k in range(len(acc_per_target))}, last_it[part])
            data['last_it'] += 1

        if it % args.print_frequency == 0:
            print(f'{lightblue(part.capitalize())}: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t',
                  f'{print_stat("MAP@3", topk_meter.val, topk_meter.avg, 4)}\t',
                  f'{print_stat("Acc",  acc_meter.val,  acc_meter.avg, 4)}\t')

        if args.niter_in_epoch > 0 and it % args.niter_in_epoch == 0 and it > 0:
            break 
                
    saver.stop()  

    print(f' * \n'
          f' * Epoch {epoch} {red(part.capitalize())}:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f'Acc  {acc_meter.avg:.4f}\t'
          f' *\t\n')

    if part != 'train':
        writer.add_scalar(f'Metrics/{part}/loss',   loss_meter.avg,  data['last_it'])
        writer.add_scalar(f'Metrics/{part}/acc',    acc_meter.avg,   data['last_it'])
        writer.add_scalar(f'Metrics/{part}/MAP@3',  topk_meter.avg,  data['last_it'])

    return loss_meter.avg


def npz_per_item(data, path, args):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys

    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    np.savez_compressed(path, **data)


def tensor_to_np_recursive(data):

    if isinstance(data, torch.Tensor): 
        return data.detach().cpu().numpy() 
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = tensor_to_np_recursive(data[k])

        return data

    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = tensor_to_np_recursive(data[i])

        return data
    else:
        return data


def npz_per_batch(data, save_dir, args, iteration):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys

    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    data = tensor_to_np_recursive(data)
    path = f'{save_dir}/{iteration}.npz'

    np.savez_compressed(path, **data)


class Saver(object):
    
    def __init__(self, args, save_fn, tq_maxsize = 5, clean_dir=True):
        super(Saver, self).__init__()
        self.args = args

        self.save_dir = args.dump_path
        self.need_save = False
        if 'save_driver' in args and args.save_driver is not None:
            

            if clean_dir and os.path.exists(args.dump_path):
                import shutil
                shutil.rmtree(args.dump_path) 

            os.makedirs(args.dump_path, exist_ok=True)

            self.tq = TaskQueue(maxsize=args.batch_size * 2, num_workers=5, verbosity=0) 

            self.save_fn = save_fn
            self.need_save = True

    def maybe_save(self, iteration, **kwargs):
        if self.need_save:
            self.tq.add_task(self.save_fn, kwargs, save_dir=self.save_dir, args=self.args, iteration=iteration)  

    def stop(self):
        if self.need_save:
            self.tq.stop_()

# def saver(args, names, x_, y_, output):
#     if 'save_driver' in args and args.save_driver is not None:

#         names =

#         y    = y_.detach().cpu().numpy() 
#         pred = output.detach().cpu().numpy()

#          tq.add_task(npz_per_item, {
#                 # 'input':  _x.detach().cpu().numpy(),
#                 # 'target': y,
#                 'pred':   pred,#.detach().cpu().numpy(),
#                 'name':   str(name),
#             }, path=f'{args.dump_path}/{name}.png', args=args)  


#         for num, (name, x) in enumerate(zip(names, x_)):
#             y = [y[num].detach().cpu().numpy() for y in y_]
#             pred = [o[num].detach().cpu().numpy() for o in output]

#             tq.add_task(npz_per_item, {
#                 # 'input':  _x.detach().cpu().numpy(),
#                 # 'target': y,
#                 'pred':   pred,#.detach().cpu().numpy(),
#                 'name':   str(name),
#             }, path=f'{args.dump_path}/{name}.png', args=args)  
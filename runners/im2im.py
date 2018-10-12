import importlib
import sys
import os.path
import time
import torch
import tqdm
import torchvision.utils as vutils
from runners.common import AverageMeter, print_stat
from PIL import Image
from huepy import lightblue, cyan

def get_args(parser):
  parser.add('--log_frequency_loss', type=int, default=50)
  parser.add('--log_frequency_images', type=int, default=1000)

  parser.add('--overwrite_images', action='store_true')
  parser.add('--niter_in_epoch', type=int, default=0)
  parser.add('--mask_it', action='store_true')

  return parser

def save_img_pil(img_np, save_path):
    Image.fromarray(img_np).save(save_path, quality=100, optimize=True, progressive=True)


def resize(imgs, sz=256):
    return torch.nn.functional.interpolate(imgs, size=sz)

last_it = dict(train=0, val=0, test=0)

def run_epoch(dataloader, model, criterion, optimizer, epoch, args, part='train'):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter     = AverageMeter()
                
    writer = run_epoch.writer
    outputs = []

    end = time.time()
    

    for it, (names, x_, y_) in enumerate(dataloader):
        
        batch_size = x_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)

        x, y = x_.cuda(non_blocking=True), y_.cuda(non_blocking=True)

        output = model(x)      

        loss = criterion(output, y)
    
        if part=='train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        loss_meter.update(loss.item(), x.shape[0])
        writer.add_scalar(f'Loss', loss_meter.val,  last_it[part])
        last_it[part] += 1
        if it % args.log_frequency_loss == 0:
            print(f'{lightblue(part.capitalize())}: [{cyan(epoch)}][{cyan(it)}/{cyan(len(dataloader) if args.niter_in_epoch <= 0 else args.niter_in_epoch)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t')
                    

        if it % args.log_frequency_images == 0:
            writer.add_image(f'{part}_{epoch}', get_grid(x_, y_, output),  it)
              
        if args.niter_in_epoch > 0 and it % args.niter_in_epoch == 0 and it > 0:
            break          
    
    print(f' * \n'
          f' * Epoch {epoch}, {part.capitalize()} stat:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f' *\t\n')

    return loss_meter.avg



def get_grid(x, y, output):
    num_img = min(x.shape[0], 4)
    imgs = resize(torch.cat([x[:num_img].detach().cpu(), y[:num_img].detach().cpu(), output[num_img].detach().cpu()]))
    x = vutils.make_grid(imgs, nrow = num_img)
    
    return x













       # writer.add_image('target', target[:3], it)
            # writer.add_image('s', s.detach().cpu().numpy().transpose( 2, 0, 1 ), it)

            # for dd, (name, pred, target, s) in enumerate(zip(names, output, y_, stickman)):
            #     # path = f'{args.experiment_dir}/samples'
            #     path = f'out/{args.config_name}'
            #     if not os.path.exists(path):
            #         os.makedirs(path)

            #     # pred

               

            #     # print (pred.shape)
            #     pred   = (pred[:3].detach().cpu().numpy().transpose( 1, 2, 0 )   * 255).astype(np.uint8)
            #     target = (target[:3].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            #     # print(s.max())
            #     st = (s[:3].detach().cpu().numpy()).astype(np.uint8)

            #     basename = os.path.basename(name)

            #     st_name  = f'{path}/{part}_{epoch}_{dd}_x.png'
            #     target_name = f'{path}/{part}_{epoch}_{dd}_gt.png'
            #     pred_name   = f'{path}/{part}_{epoch}_{dd}_pred.png'

            #     # if not os.path.exists(st_name):
            #     save_img_pil(st, st_name)
            #     # if not os.path.exists(y_name):
            #     save_img_pil(target, target_name)

            #     save_img_pil(pred, pred_name)



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
from torch.autograd import Variable
from torch.nn.functional import softmax
from .common import AverageMeter, accuracy

alpha = 0.1
# dbeta = beta.Beta(torch.Tensor([alpha]), torch.Tensor([1 + alpha]))

def run_epoch_train(dataloader, model, criterion, optimizer, epoch, print_freq = 1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
            
    outputs = []

    end = time.time()
    # for it, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader, dataloader)):
        

    #     # Measure data loading time
    #     data_time.update(time.time() - end)

    #     x1, x2 = x2.cuda(async=True), x2.cuda(async=True)
    #     y1, y2 = y1.cuda(async=True), y2.cuda(async=True)
        
        # lam = dbeta.sample( (x1.shape[0], 1, 1, 1))
    for it, (_, x_, y_) in enumerate(dataloader):
        
        batch_size = x_.shape[0]  

        # Measure data loading time
        data_time.update(time.time() - end)

        # print(batch_size / 2)
        x1, x2 = x_[:batch_size // 2].cuda(async=True), x_[batch_size // 2:].cuda(async=True)
        y1, y2 = y_[:batch_size // 2].cuda(async=True), y_[batch_size // 2:].cuda(async=True)
        # print (torch.norm((x1 - x2)))
        
        lam = torch.from_numpy(np.random.beta(alpha+1, alpha, [batch_size // 2,1,1,1]).astype(np.float32)).cuda(async=True)

        x_var  = Variable(x1 * lam + x2 * (1 - lam))
        y_var = Variable(y1)

        output = model(x_var)

        y_var = y_var.type(torch.cuda.LongTensor)
        losses_ = [criterion(output[i], y_var[:, i]) for i in range(len(output))]
        loss = sum(losses_)

        losses.update(loss.data[0], x1.shape[0])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % print_freq == 0:
            print(f'Epoch: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
                  #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  
    
    print(f' * \n'
          f' * Epoch {epoch} Training:\t'
          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
          #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
          f' *\t\n')

        
def run_epoch_test(dataloader, model, criterion, epoch, print_freq = 1, need_softmax=False, need_preds=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    avg_loss = AverageMeter()
    top1 = AverageMeter()
     
    
    outputs, all_names = [], []

    end = time.time()
    for it, (names, x, y) in enumerate(dataloader):
        
        # Measure data loading time
        data_time.update(time.time() - end)

        x = x.cuda(async=True)
        y = y.cuda(async=True)
        
        x_var  = Variable(x, volatile=True)
        y_var  = Variable(y, volatile=True)

        output = model(x_var)

        y_var = y_var.type(torch.cuda.LongTensor)
        losses_ = [criterion(output[i], y_var[:, i]) for i in range(len(output))]
        loss = sum(losses_)

        if need_softmax:
            output = [softmax(o) for o in output]
            
        if need_preds:
            outputs.append([o.to_numpy() for o in output])



        avg_loss.update(loss.data[0], x.shape[0])

        # prec1 = accuracy(output.data, target, topk=(1, 5))
        top1_ = [accuracy(output[i].data, y_var[:, i].data, topk=(1,)) for i in range(len(output))]
        top1.update(sum([o[0].cpu().numpy()[0] for o in top1_]), x.shape[0])
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % print_freq == 0:
            print(f'Epoch: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  f'Loss {avg_loss.val:.4f} ({avg_loss.avg:.4f})\t'\
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t')
                  #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  
        
        all_names.append(names)

    print(f' * \n'
          f' * Epoch {epoch} Testing:\t'
          f'Loss {avg_loss.val:.4f} ({avg_loss.avg:.4f})\t'
          f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
          f' *\t\n')

    if need_preds:
        # outputs = np.concatenate(outputs)

        return loss.data[0], outputs

    return loss.data[0]


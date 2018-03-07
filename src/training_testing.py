import importlib
import sys
import random
import os.path
# from dataloader import *
import time
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.functional import softmax
# print_freq = 1

def fn(self):
    return self.cpu().data.numpy()

torch.autograd.Variable.to_numpy = fn

def run_epoch(dataloader, model, criterion, optimizer, epoch, MODE, phase, print_freq = 1, need_softmax=False, need_preds=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    
    needs_backward = (phase == 'train')
    model.train() if phase == 'train' else model.eval()
    
    outputs = []

    end = time.time()
    for it, (input, target, mask) in enumerate(dataloader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        input  = input.cuda(async=True)
        target = target.cuda(async=True)
        mask   = mask.cuda(async=True)

        input_var  = Variable(input , volatile = not needs_backward)
        target_var = Variable(target, volatile = not needs_backward)
        mask_var   = Variable(mask  , volatile = not needs_backward)

        output = model(input_var)

        if MODE == 'classification':
            target_var = target_var.type(torch.cuda.LongTensor)
            losses_ = [criterion(output[i], target_var[:, i]) for i in range(len(output))]
            loss = sum(losses_)

            if need_softmax:
                output = softmax(x)
            
            if need_preds:
                outputs.append([x.to_numpy() for x in output])
        else:
            if MODE == 'regression_masked':
                output = output * mask_var
                target_var = target_var * mask_var

            loss = criterion(output, target_var)

        # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target,
#                                 topk=(1, min(5, args.n_classes)))
        losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
    
        if needs_backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
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
          f' * Epoch {epoch} {phase}:\t'
          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
          #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
          f' *\t\n')

    if need_preds:
        outputs = np.concatenate(outputs)

        return outputs
        



# def train(train_loader, model, criterion, optimizer, epoch, MODE):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (input, target, mask) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)

#         input = input.cuda(async=True)
#         target = target.cuda(async=True)
#         mask = mask.cuda(async=True)

#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
#         mask_var = torch.autograd.Variable(mask)

#         output = model(input_var)

#         if MODE == 'classification':
#             target_var = target_var.type(torch.cuda.LongTensor)
#             losses_ = [criterion(output[i], target_var[:, i]) for i in range(len(output))]
#             loss = sum(losses_)
#         else:
#             if MODE == 'regression_masked':
#                 output = output * mask_var
#                 target_var = target_var * mask_var
#             loss = criterion(output, target_var)

#         # measure accuracy and record loss
# #         prec1, prec5 = accuracy(output.data, target,
# #                                 topk=(1, min(5, args.n_classes)))
#         losses.update(loss.data[0], input.size(0))
# #         top1.update(prec1[0], input.size(0))
# #         top5.update(prec5[0], input.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
#                   ''.format(
#                       epoch, i, len(train_loader), batch_time=batch_time,
#                       data_time=data_time, loss=losses, top1=top1, top5=top5))

#     print(' * \n'
#           ' * Epoch {0} training:\t'
#           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#           #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#           #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
#           ' *\t'.format(
#               epoch, loss=losses, top1=top1, top5=top5))



def validate(val_loader, model, criterion, MODE):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    outputs = []
    end = time.time()
    for i, (input, target, mask) in enumerate(val_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        mask = mask.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        mask_var = torch.autograd.Variable(mask, volatile=True)

        # compute output
#         print(input_var.shape)
        output = model(input_var)
        if MODE == 'classification':
            target_var = target_var.type(torch.cuda.LongTensor)
            losses_ = [criterion(output[i], target_var[:, i]) for i in range(len(output))]
            loss = sum(losses_)

            outputs.append([x.cpu().data.numpy() for x in output])
        else:
            if MODE == 'regression_masked':
                output *= mask_var
                target_var *= mask_var

            loss = criterion(output, target_var)
            outputs.append(output.cpu().data.numpy())

        # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, min(5, args.n_classes)))
        losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  ''.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print(' *')
    print(' * Testing:\tLoss {loss.avg:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(loss=losses, top1=top1, top5=top5))
    print(' *')

    return outputs, losses.avg


def test(val_loader, model, criterion, mode, need_softmax=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    outputs = []
    end = time.time()
    for i, (input, target, mask) in enumerate(val_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
#         print(input_var.shape)
        output = model(input_var)
        if mode == 'classification':
            target_var = target_var.type(torch.cuda.LongTensor)
            losses_ = [criterion(output[i], target_var[:, i]) for i in range(len(output))]
            loss = sum(losses_)

            if need_softmax:
                outputs.append([torch.nn.functional.softmax(x).cpu().data.numpy() for x in output])
            else:
                outputs.append([x.cpu().data.numpy() for x in output])
        else:
            loss = criterion(output, target_var)
            outputs.append(output.cpu().data.numpy())

        # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, min(5, args.n_classes)))
        losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  ''.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print(' *')
    print(' * Testing:\tLoss {loss.avg:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(loss=losses, top1=top1, top5=top5))
    print(' *')

    outputs = np.concatenate(outputs)

    return outputs, losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from huepy import red
from torch import nn
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from contrib.criterions.focal_loss import *
from contrib.criterions.huber_loss import *
from contrib.criterions.lovasz_losses import lovasz_hinge, lovasz_loss_ignore_empty


import numpy as np
import os 
import sys
import torch
import torch.nn.functional as F
import torch.nn.functional as fnn

# import encoding.nn

# from models.DataParallelCriterion import DataParallelCriterion

def get_criterion(name, args, **kwargs):

    criterion_args = {}
    if args.criterion_args != '':
        for entry in args.criterion_args.split("^"):
            k, v = entry.split('=')
            criterion_args[k] = eval(v)

    print(criterion_args)
    if name in sys.modules[__name__].__dict__:
        criterion = sys.modules[__name__].__dict__[name](**criterion_args)
    elif name in torch.nn.modules.__dict__:
        criterion = torch.nn.modules.__dict__[name](**criterion_args)
    else:
        assert False, red(f"Cannot find loss with name {name}")

    # if args.use_all_gpus and args.parallel_criterion:
    #     criterion = DataParallelCriterion(criterion)

    return criterion

# -------------------------------------------------
# ----------------- Lap1Loss  ---------------------
# -------------------------------------------------

class LapL1Loss(_Loss):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapL1Loss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp( (x - size // 2)**2 / (-2 * sigma**2) )**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :], requires_grad=False)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


# -------------------------------------------------
# ------------- L1CosineLoss  ---------------
# -------------------------------------------------

class L1CosineLoss(_Loss):

    def __init__(self, l=1):
        super(L1CosineLoss, self).__init__()
        self.l = l

    def __call__(self, input, target):
        return torch.nn.functional.l1_loss(input, target) + 1 - (self.l * torch.nn.functional.cosine_similarity(input, target, dim=1)).mean()


    def cuda(self):
        return self



# -------------------------------------------------
# ------------- MultiHeadCriterion  ---------------
# -------------------------------------------------



class MultiHeadCriterion(_Loss):
    '''
        For a number of "1-of-K" tasks. 

    '''
    def __init__(self, weight=None, losses='auto', size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MultiHeadCriterion, self).__init__(size_average, reduce, reduction)
        
        self.weight = weight
        if weight is not None:
            self.weight = weight.cuda()

        self.losses = losses
        # if losses != 'auto': 


    def forward(self, inputs, targets):
        '''
            let N be a number of different tasks 

            `target` is a list of size N, where each element is either a vector of size `B` (then is is multi-class task)
                or of size `B` x P_i, where P_i is the number of labels (in multi-label task e.g. tagging)

            `target` is a list of size N, where each element is of size B x P_i for i = 1 ... N
        '''

        losses = {}
        for i, (input, target) in enumerate(zip(inputs, targets)):
            # print(inp.shape)
            if target is None:
                continue

            # print(tar.shape, inp.shape)
            if (len(target.shape) == 1 or target.shape[1] == 1) and input.shape[1] > 1:
                loss = nn.CrossEntropyLoss(weight=self.weight)
                losses[i] = loss(input, target)
            else: 
                print(' !!!!! ' )
                # loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(0.00636))
                # loss = nn.BCEWithLogitsLoss()
                loss = nn.L1Loss()


                losses[i] = loss(input, target.float())

        # loss = sum(losses) / len(losses)

        return losses


# -------------------------------------------------
# ------------- Hungarian -------------------------
# -------------------------------------------------
from pdb import set_trace as bp
import scipy.optimize 

class Hungarian(_Loss):

    def __init__(self, l=1):
        super().__init__()

        

        self.l = l

    def cdist(self, input1, input2, norm = 2):
        input1 = input1.unsqueeze(2)
        input2 = input2.unsqueeze(1)
        
        if norm == 2:
            return (input1 - input2).pow(2).mean(3)
        
        if norm == 1:
            return torch.abs(input1 - input2).mean(3)

    def __call__(self, input, target):
        '''
            input is a list of predictions 
            target is a list of targets 
        '''

        # 1. compute distance 

        

        inputs = torch.cat([x.unsqueeze(1) for x in input], 1) # B x num(vecs) x len(vec)
        
        dist_mat = self.cdist(inputs, target[0], self.l)


        # 2. Get assignment
        loss = 0
            
        dm = dist_mat.detach().cpu().numpy()

        for i in range(inputs.shape[0]):
            res = scipy.optimize.linear_sum_assignment(dm[i, :target[1][i], :target[1][i] ])

            loss += sum([dist_mat[i, x, y] for x,y in zip(res[0], res[1])])

        return {'all': loss}


    def cuda(self):
        return self


class CriterionList(_Loss):
    def __init__(self, criterions=None, weights=None):
        super(CriterionList, self).__init__()
        
        self.criterions = [
            # LovaszSoftmaxFlat(),
            # nn.CrossEntropyLoss(), 
            # nn.L1Loss(),
        ]

        if weights is None:
            self.weights = [1] * len(self.criterions)
        else:
            self.weights = weights

    def forward(self, inputs, targets):

        lens = [len(x) for x in [inputs, targets, self.weights, self.criterions]]

        assert all([x == lens[0] for x in lens]), print(lens)
        

        losses = {}
        for i, (input, target, weight, criterion) in enumerate(zip(inputs, targets, self.weights, self.criterions)):
            
            if target is None or weight == 0:
                continue

            losses[i] = criterion(input, target) * weight
           
        return losses




class ColorRecognitionLoss(_Loss):
    def __init__(self, num_colors_weight=0.2, need_sigmoid=True):
        super().__init__()
        
        self.ce =  nn.CrossEntropyLoss()
        self.color_loss = Hungarian()

        self.num_colors_weight = num_colors_weight
        self.need_sigmoid = need_sigmoid

    def forward(self, inputs, targets):


        losses = {}

        losses['color'] = self.color_loss([torch.sigmoid(x) if self.need_sigmoid else x for x in inputs[:-1]], targets)['all'] 
        losses['num_colors'] = self.ce(inputs[-1], targets[1] - 1) * self.num_colors_weight
        
        return losses









def symmetric_lovasz(outputs, targets):
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2


def symmetric_lovasz_ignore_empty(outputs, targets, truth_image):
    return (lovasz_loss_ignore_empty(outputs, targets, truth_image) +
            lovasz_loss_ignore_empty(-outputs, 1 - targets, truth_image)) / 2


class DeepSupervisedCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(DeepSupervisedCriterion, self).__init__(size_average, reduce, reduction)

        self.is_average = True 

    def forward(self, input, targets):

        pred_pixel_lvl, preds_middle, pred_image_lvl = input 

        total_loss = 0
        for ti in range(targets.shape[1]):
            target = targets[:, ti, ...]
            target_image_lvl = (target.sum(dim=(1, 2)) > 0).float()

            loss_image = fnn.binary_cross_entropy_with_logits(pred_image_lvl, target_image_lvl, reduce=self.is_average)
            loss_pixel = 0
            for pred_m in preds_middle:
                loss_pixel += symmetric_lovasz_ignore_empty(pred_m.squeeze(1), target, target_image_lvl)
                # loss_pixel += fnn.binary_cross_entropy_with_logits(pred_m.squeeze(1), target)

            # loss = fnn.binary_cross_entropy_with_logits(pred_pixel_lvl.squeeze(1), target)    
            loss = symmetric_lovasz(pred_pixel_lvl.squeeze(1), target)

            total_loss += 0.05 * loss_image + 0.1 * loss_pixel + 1 * loss

        return total_loss

class LossBinaryDice(nn.Module):
   def __init__(self, dice_weight=1):
       super(LossBinaryDice, self).__init__()
       self.nll_loss = nn.BCEWithLogitsLoss()
       self.dice_weight = dice_weight

   def forward(self, outputs, targets):
       loss = self.nll_loss(outputs, targets)

       if self.dice_weight:
           smooth = 1e-5
           target = (targets == 1.0).float()
           prediction = F.sigmoid(outputs)
           dice_part = 1 - 2 * (torch.sum(prediction * target) + smooth) / \
                          (torch.sum(prediction) + torch.sum(target) + smooth)


           loss += self.dice_weight * dice_part

       return loss


class DeepSupervisedBCECriterion(_Loss):
    '''
        image level loss + pixel level loss + additional supervision 
    '''
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(DeepSupervisedBCECriterion, self).__init__(size_average, reduce, reduction)

        self.is_average = True 

    def forward(self, input, targets):

        pred_pixel_lvl, preds_middle, pred_image_lvl = input 
        target_pixel_lvl, target_image_lvl = targets 

        # print(pred_image_lvl, target_image_lvl)
        # print(target_image_lvl, target_pixel_lvl.max(), target_pixel_lvl.mean())
        # print(pred_image_lvl.shape, target_image_lvl.shape)
        loss_image_lvl = fnn.binary_cross_entropy_with_logits(pred_image_lvl, target_image_lvl)


        loss_middle = torch.tensor(0, dtype=torch.float32, device=pred_pixel_lvl.device) 
        
        # m = target_image_lvl > 0

        # if m.sum().item() > 0:
        #     for pred_m in preds_middle:
        #         loss_middle += 0.4 * fnn.binary_cross_entropy_with_logits(pred_m[m], target_pixel_lvl[m]) + dice_loss(torch.sigmoid(pred_m[m]), target_pixel_lvl[m])
            
        pixel_lvl_loss = 0.4 * fnn.binary_cross_entropy_with_logits(pred_pixel_lvl, target_pixel_lvl) 

        # if m.sum().item() > 0:
        #     pixel_lvl_loss += dice_loss(torch.sigmoid(pred_pixel_lvl[m]), target_pixel_lvl[m])
            
        total_loss = 1. * loss_image_lvl + 1 * pixel_lvl_loss + 0.1 * loss_middle

        sep_losses = {
            'image_lvl': loss_image_lvl,
            'pixel_lvl_loss': pixel_lvl_loss,
            'loss_middle': loss_middle
        }
        
        return total_loss, sep_losses


class DeepSupervisedBCEPretrainCriterion(_Loss):
    '''
        image level loss + pixel level loss + additional supervision 
    '''
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(DeepSupervisedBCEPretrainCriterion, self).__init__(size_average, reduce, reduction)

        self.is_average = True 

    def forward(self, input, targets):

        pred_pixel_lvl, preds_middle, pred_image_lvl = input 
        target_pixel_lvl, target_image_lvl = targets 

        # print(pred_image_lvl, target_image_lvl)
        # print(target_image_lvl, target_pixel_lvl.max(), target_pixel_lvl.mean())
        # print(pred_image_lvl.shape, target_image_lvl.shape)
        loss_image_lvl = fnn.binary_cross_entropy_with_logits(pred_image_lvl, target_image_lvl)


        # loss_middle = torch.tensor(0, dtype=torch.float32, device=pred_pixel_lvl.device) 
        
        # m = target_image_lvl > 0

        # if m.sum().item() > 0:
        #     for pred_m in preds_middle:
        #         loss_middle += 0.4 * fnn.binary_cross_entropy_with_logits(pred_m[m], target_pixel_lvl[m]) + dice_loss(torch.sigmoid(pred_m[m]), target_pixel_lvl[m])
            
        # pixel_lvl_loss = 0.4 * fnn.binary_cross_entropy_with_logits(pred_pixel_lvl, target_pixel_lvl) 

        # if m.sum().item() > 0:
        #     pixel_lvl_loss += dice_loss(torch.sigmoid(pred_pixel_lvl[m]), target_pixel_lvl[m])
            
        total_loss = 1. * loss_image_lvl # + 1 * pixel_lvl_loss + 0.1 * loss_middle

        sep_losses = {
            'image_lvl': loss_image_lvl,
            # 'pixel_lvl_loss': pixel_lvl_loss,
            # 'loss_middle': loss_middle
        }
        
        return total_loss, sep_losses




eps = 1e-3
def dice_loss(preds, trues, weight=None, is_average=False):
    b, c = preds.shape[0], preds.shape[1]
    
    preds = preds.view(b, c, -1)
    trues = trues.view(b, c, -1)

    intersection = (preds * trues).sum(2)
    scores = 1 - (2. * intersection + eps) / (preds.sum(2) + trues.sum(2) + eps)

    return scores.mean()
        


import torch as th                                                                 

class NLL_OHEM(_Loss):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSotmax() """                                             
                                                                                   
    def __init__(self, threshold=0.22):      
        super(NLL_OHEM, self).__init__(None, True)                                 
        self.threshold = threshold                                                         
        self.loss = nn.CrossEntropyLoss(reduction='none')   

    def forward(self, x, y):                                           
        # if ratio is not None:                                                      
        #     self.ratio = ratio  


        losses = self.loss(x[0], y[0])

        mask = losses > self.threshold 

        # print(losses[mask].shape)
        return losses[mask].mean()


#################################3


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmaxFlat(_Loss):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSotmax() """                                             
                                                                                   
    def __init__(self, threshold=0.22):      
        super(LovaszSoftmaxFlat, self).__init__(None, True)                                 
        self.threshold = threshold                                                         
        # self.loss = nn.CrossEntropyLoss(reduction='none')   

    def forward(self, x, y):   
        return lovasz_softmax_flat(torch.softmax(x, 1), y)                                        


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    only_present: average only on classes present in ground truth
    """
    
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue

        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return sum(losses)/len(losses)




class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1) 

import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
from collections import OrderedDict
from os.path import expanduser

class VGGLossMix(nn.Module):
    def __init__(self, weight=0.5):
        super(VGGLossMix, self).__init__()
        self.l1 = VGGLoss()
        self.l2 = VGGLoss(net='caffe')
        self.weight = weight

    def forward(self, input, target):
        return self.l1(input, target)*self.weight + self.l2(input, target) * (1-self.weight)

        
class VGGLoss(nn.Module):
    def __init__(self, net='pytorch', normalize_grad=False):
        super(VGGLoss, self).__init__()

        self.normalize_grad=normalize_grad
        
        if net == 'pytorch':
            vgg19 = torchvision.models.vgg19(pretrained=True).features
            self.mean_ = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
            self.std_  = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

        elif net == 'caffe':
            if not os.path.exists('~/.torch/models/vgg_caffe_features.pth'):
                vgg_weights = torch.utils.model_zoo.load_url('https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth') 
                
                map = {'classifier.6.weight':u'classifier.7.weight', 'classifier.6.bias':u'classifier.7.bias'}
                vgg_weights = OrderedDict([(map[k] if k in map else k,v) for k,v in vgg_weights.items()])

                

                model = torchvision.models.vgg19()
                model.classifier = nn.Sequential(View(), *model.classifier._modules.values())
                

                model.load_state_dict(vgg_weights)
                
                vgg19 = model.features
                torch.save(vgg19, f'{expanduser("~")}/.torch/models/vgg_caffe_features.pth')

                

                self.mean_ = torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.
                self.std_   = torch.FloatTensor([1./255, 1./255, 1./255])[None, :, None, None]
            else:
                vgg19 = torch.load(f'{expanduser("~")}/.torch/models/vgg_caffe_features.pth')
        else:
            assert False

        vgg19_avg_pooling = []

        
        for weights in vgg19.parameters():
            weights.requires_grad = False

        for module in vgg19.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg19_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg19_avg_pooling.append(module)
        
        vgg19_avg_pooling = nn.Sequential(*vgg19_avg_pooling)

        print(vgg19_avg_pooling)
        self.vgg19 = vgg19_avg_pooling
        
        

    def normalize_inputs(self, x):
        return (x - self.mean_.cuda()) / self.std_.cuda()


    def forward(self, input, target):
        loss = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)
        for layer in self.vgg19[:30]:

            features_input  = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':

                if self.normalize_grad:
                    pass
                else:
                    loss = loss + F.l1_loss(features_input, features_target)

        return loss




class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, input):
        loss = F.softmax(input, dim=1)*F.log_softmax(input, dim=1)
        return loss.mean()

class DummyCriterion(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyCriterion, self).__init__()

    def forward(self, input, target):
        return torch.FloatTensor([0]).cuda()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        self.epsilon = 1e-6

    def forward(self, input):
        loss = torch.pow(input[:, :, 1:, :-1] - input[:, :, :-1, :-1], 2) + torch.pow(input[:, :, :-1, 1:] - input[:, :, :-1, :-1], 2)
        loss = torch.sqrt(loss + self.epsilon).mean()
        return loss


import torch
import torch.nn.functional as F
import numpy as np
import pdb

def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    #assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."

    probs=F.softmax(input)
    #probs = input
    num=probs*torch.cat((1-target, target), 1) #b,c,h,w--p*g
    num=torch.sum(num,dim=3)
    num=torch.sum(num,dim=2)#b,c

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)
    den1=torch.sum(den1,dim=2)#b,c,1,1

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)
    den2=torch.sum(den2,dim=2)#b,c,1,1

    dice=2*(num/(den1+den2+1e-7))
    dice_eso=dice[:,1]#we ignore bg dice val, and take the fg

    loss=1-torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
    dc = (1-loss).cpu()
    acc = probs.cpu().max(1)[1].eq(target.squeeze(1).long().cpu()).sum().float() / target.cpu().view(-1).size(0)

    return loss, acc, dc


def dice_loss_bak(output, target):
    """Compute dice among **positive** labels to avoid unbalance.
    Arguments:
        output: [batch_size * height * width,  2 ] (torch.cuda.FloatTensor)
        target: [batch_size * height * width, (1)] (torch.cuda.FloatTensor)
    Returns:
        tuple contains:
        + dice loss: for back-propagation
        + accuracy: (predtrue - true) / true
        + dice overlap:  2 * predtrue * true / (predtrue - true) * 100
        + predtrue: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
        + true: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
    """

    pdb.set_trace()

    predict = (output.max(1)[1]).float()  # {0, 1}. 0 for the original output, 1 for the binary mask
    target = (target.squeeze(1)).float()

    # Loss
    intersection = torch.sum(predict * target, 0)
    union = torch.sum(predict * target, 0) + torch.sum(target * target, 0)
    dice = 2.0 * intersection / (union + 1e-7)
    loss = 1 - dice

    # Overlap
    predtrue = predict.eq(1).float().data    # FloatTensor 0.0 / 1.0
    true = target.data                  # FloatTensor 0.0 / 1.0
    overlap = 2 * (predtrue * true).sum() / (predtrue.sum() + true.sum() + 1e-7) * 100

    # Accuracy
    acc = predtrue.eq(true).float().mean()
    #return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7), acc, overlap, predtrue, true
    return loss, acc, overlap
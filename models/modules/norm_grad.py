import torch

class NormGrad(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        # print()
        grad_input = grad_output.clone()
        grad_input = torch.nn.functional.normalize(grad_input)

        print(torch.norm(grad_output), torch.norm(grad_input))
        return grad_input

normgrad = NormGrad.apply 

class NormGradModule(torch.nn.Module):
   def __init__(self, in_features):
       super(NormGradModule, self).__init__()
       

   def forward(self, x):
       out = normgrad(x)
       return out 

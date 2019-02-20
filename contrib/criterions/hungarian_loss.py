from pdb import set_trace as bp
import scipy.optimize 
import torch 
from torch.nn.modules.loss import _Loss

class HungarianLoss(_Loss):

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
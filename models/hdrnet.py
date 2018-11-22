import torch 
import torch.nn as nn
from models.model import get_abstract_net, get_model_args, get_abstract_native_transform
from models.common import Identity

@get_model_args
def get_args(parser):
    parser.add('--pw_guide', action='store_bool')
    parser.add('--use_bn', default=False, action='store_bool')
    return parser


@get_abstract_net
def get_net(args):

    model = RetouchGenerator(args.pw_guide, use_bn=args.use_bn)
    return model 

@get_abstract_native_transform
def get_native_transform():
    return None
    

class RetouchGenerator(nn.Module):
    def __init__(self, pw_guide=False, use_bn=False):
        super(RetouchGenerator, self).__init__()

        self.pointwise_guide = pw_guide

        ## Define layers as described in the HDRNet architecture
        # Activation
        self.activate = nn.ReLU(inplace=True)


        # Low-level layers (S)
        self.ll_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.ll_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.ll_bn2   = nn.BatchNorm2d(num_features=16) if use_bn else Identity()
        self.ll_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.ll_bn3   = nn.BatchNorm2d(num_features=32) if use_bn else Identity()
        self.ll_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.ll_bn4   = nn.BatchNorm2d(num_features=64) if use_bn else Identity()

        # Local features layers (L)
        self.lf_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lf_bn1   = nn.BatchNorm2d(num_features=64) if use_bn else Identity()
        self.lf_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        # Global features layers (G)
        self.gf_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.gf_bn1   = nn.BatchNorm2d(num_features=64) if use_bn else Identity()
        self.gf_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.gf_bn2   = nn.BatchNorm2d(num_features=64) if use_bn else Identity()
        self.gf_fc1 = nn.Linear(1024, 256)
        self.gf_fc2 = nn.Linear(256, 128)
        self.gf_fc3 = nn.Linear(128, 64)

        # Linear prediction (Pointwise layer)
        self.pred_conv = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1, stride=1)

        # Guidance map auxilary parameters
        self.pw_mat = nn.Parameter(torch.eye(3) + torch.randn(1)*1e-4, requires_grad=True)
        self.pw_bias = nn.Parameter(torch.eye(1), requires_grad=True)
        self.pw_bias_tag = nn.Parameter(torch.zeros(3,1), requires_grad=True)

        self.rho_a = nn.Parameter(torch.ones(16,3), requires_grad=True)
        self.rho_t = nn.Parameter(torch.rand(16,3), requires_grad=True)

        # Pointwise guidance map
        self.pw_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1)
        self.pw_batchnorm = nn.BatchNorm2d(16)
        self.pw_conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1)
        self.pw_activate = nn.Sigmoid()


    def forward(self, high_res, low_res):
        bg = self.create_bilateral(low_res)
        guide = self.create_guide(high_res)
        #output = self.slice_and_assemble(bg, guide, high_res)
        output = self.slice_and_assemble_exp(bg, guide, high_res)
        return output

    def create_bilateral(self, low_res):
        ## TODO: Add batch normalization
        # Low-level
        ll =             self.activate(self.ll_conv1(low_res))
        ll = self.activate(self.ll_bn2(self.ll_conv2(ll)))
        ll = self.activate(self.ll_bn3(self.ll_conv3(ll)))
        ll = self.activate(self.ll_bn4(self.ll_conv4(ll)))

        # Local features
        lf = self.lf_bn1(self.activate(self.lf_conv1(ll)))
        lf = self.lf_conv2(lf)  # No activation (normalization) before fusion

        # Global featuers
        gf = self.activate(self.gf_bn1(self.gf_conv1(ll)))
        gf = self.activate(self.gf_bn2(self.gf_conv2(gf)))
        gf = self.activate(self.gf_fc1(gf.view(-1, gf.shape[1]*gf.shape[2]*gf.shape[3])))
        gf = self.activate(self.gf_fc2(gf))
        gf = self.gf_fc3(gf)  # No activation (normalization) before fusion

        # Fusion
        fusion = self.activate(gf.view(-1, 64, 1, 1) + lf)

        # Bilateral Grid
        pred = self.pred_conv(fusion)
        # bilateral_grid = pred.view(-1, 12, 16, 16, 8)  # Image features as a bilateral grid
        bilateral_grid = pred.view(-1, 12, 8, 16, 16)   # unroll grid

        return bilateral_grid

    def create_guide(self, high_res):
        if not self.pointwise_guide:
            guide = high_res.view(high_res.shape[0], 3, -1)                             # (nbatch, nchannel, w*h)
            guide = torch.matmul(self.pw_mat, guide)
            guide = guide + self.pw_bias_tag
            guide = self.activate(guide.unsqueeze(1) - self.rho_t.view([16, 3, 1]))     # broadcasting to (nbatch, 16, nchannel, w*h)
            guide = guide.permute(0,3,1,2) * self.rho_a                                 # (nbatch, w*h, 16, nchannel)
            guide = guide.sum(3).sum(2) + self.pw_bias
            guide = guide.view(high_res.shape[0],high_res.shape[2],high_res.shape[3])   # return to original shape
        else:
            guide = self.pw_conv1(high_res)
            guide = self.pw_batchnorm(guide)
            guide = self.pw_conv2(guide)
            guide = guide.squeeze()
            guide = self.pw_activate(guide)

        return guide

    def slice_and_assemble(self, bg, guide, high_res):
        device = bg.device

        bg = bg.permute(0, 1, 3, 4, 2)
        # clip guide to [-1,1] to comply with 'grid_sample'
        guide = (guide / guide.max(2)[0].max(1)[0].unsqueeze(1).unsqueeze(2))*2 - 1
        bs, gh, gw = guide.shape

        output = torch.zeros((bs, 3, gh, gw)).to(device)

        # create xy meshgrid for bilateral grid slicing
        x = torch.linspace(-1, 1, gw)
        y = torch.linspace(-1, 1, gh)
        x_t = x.repeat([gh,1]).to(device)
        y_t = y.view(-1,1).repeat([1, gw]).to(device)
        xy = torch.cat([x_t.unsqueeze(2), y_t.unsqueeze(2)], dim=2)

        for b in range(0, bs):
            guide_aug = torch.cat([xy, guide[b].unsqueeze(2)], dim=2).unsqueeze(0).unsqueeze(0)         # augment meshgrid with z dimension (1,out_d,out_h,out_w,idx)
            slice = nn.functional.grid_sample(bg[b:b+1], guide_aug)                                     # slice bilateral grid

            # assemble output
            output[b,0,:,:] = slice[0,3,0] + slice[0,0,0]*high_res[b,0,:,:] + slice[0,1,0]*high_res[b,1,:,:] + slice[0,2,0]*high_res[b,2,:,:]
            output[b,1,:,:] = slice[0,7,0] + slice[0,4,0]*high_res[b,0,:,:] + slice[0,5,0]*high_res[b,1,:,:] + slice[0,6,0]*high_res[b,2,:,:]
            output[b,2,:,:] = slice[0,11,0] + slice[0,8,0]*high_res[b,0,:,:] + slice[0,9,0]*high_res[b,1,:,:] + slice[0,10,0]*high_res[b,2,:,:]

        return output

    def slice_and_assemble_exp(self, grid, guide, input):
        bs,c,gd,gw,gh = grid.shape
        _,ci,h,w = input.shape

        xx = torch.arange(0, w, dtype=torch.float).cuda().view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h, dtype=torch.float).cuda().view(-1, 1).repeat(1, w)
        gx = ((xx + 0.5) / w) * gw
        gy = ((yy + 0.5) / h) * gh

        # print(gx.type())
        gz = torch.clamp(guide, 0.0, 1.0) * gd
        fx = torch.clamp(torch.floor(gx - 0.5), min=0)
        fy = torch.clamp(torch.floor(gy - 0.5), min=0)
        fz = torch.clamp(torch.floor(gz - 0.5), min=0)
        wx = gx - 0.5 - fx
        wy = gy - 0.5 - fy
        wx = wx.unsqueeze(0).unsqueeze(0)
        wy = wy.unsqueeze(0).unsqueeze(0)
        wz = torch.abs(gz - 0.5 - fz)
        wz = wz.unsqueeze(1)
        fx = fx.long().unsqueeze(0).unsqueeze(0)
        fy = fy.long().unsqueeze(0).unsqueeze(0)
        fz = fz.long()
        cx = torch.clamp(fx + 1, max=gw - 1);
        cy = torch.clamp(fy + 1, max=gh - 1);
        cz = torch.clamp(fz + 1, max=gd - 1)
        fz = fz.view(bs, 1, h, w)
        cz = cz.view(bs, 1, h, w)
        batch_idx = torch.arange(bs).view(bs, 1, 1, 1).long().cuda()
        out = []
        co = c // (ci + 1)
        for c_ in range(co):
            c_idx = torch.arange((ci + 1) * c_, (ci + 1) * (c_ + 1)).view( \
                1, ci + 1, 1, 1).long().cuda()
            a = grid[batch_idx, c_idx, fz, fy, fx] * (1 - wx) * (1 - wy) * (1 - wz) + \
                grid[batch_idx, c_idx, cz, fy, fx] * (1 - wx) * (1 - wy) * (wz) + \
                grid[batch_idx, c_idx, fz, cy, fx] * (1 - wx) * (wy) * (1 - wz) + \
                grid[batch_idx, c_idx, cz, cy, fx] * (1 - wx) * (wy) * (wz) + \
                grid[batch_idx, c_idx, fz, fy, cx] * (wx) * (1 - wy) * (1 - wz) + \
                grid[batch_idx, c_idx, cz, fy, cx] * (wx) * (1 - wy) * (wz) + \
                grid[batch_idx, c_idx, fz, cy, cx] * (wx) * (wy) * (1 - wz) + \
                grid[batch_idx, c_idx, cz, cy, cx] * (wx) * (wy) * (wz)
            o = torch.sum(a[:, :-1, ...] * input, 1) + a[:, -1, ...]
            out.append(o.unsqueeze(1))
        out = torch.cat(out, 1)

        return out
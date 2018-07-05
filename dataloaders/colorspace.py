from __future__ import division

import torch
import numpy

def rgb2lab(img_in):
    '''
    transformation froim RGB to Lab.
    Input:
        img_in: (bsz, c, h, w) tensor of rgb image batch with value range (-1, 1)
    Output:
        img_out: (bsz, c, h, w) tensof of Lab image batch.
    '''
    bsz, c, h, w = img_in.size()
    rgb = img_in
    
#     rgb = (img_in + 1 + 1e-5) * 0.5
    # convert to XYZ space
    # rgb = torch.where(rgb>0.04045, torch.pow((rgb+0.055)/1.055, 2.4), rgb/12.92)
    # x = (rgb[:,0]*0.4124 + rgb[:,1]*0.3576 + rgb[:,2]*0.1805) / 0.95047
    # y = (rgb[:,0]*0.2126 + rgb[:,1]*0.7152 + rgb[:,2]*0.0722) / 1.
    # z = (rgb[:,0]*0.0193 + rgb[:,1]*0.1192 + rgb[:,2]*0.9505) / 1.08883
    # xyz = torch.stack([x,y,z], dim=1)
    # xyz = torch.where(xyz>0.008856, torch.pow(xyz, 1./3), 7.787*xyz + 16./116)
    
    m1 = (rgb > 0.04045).float()
    rgb = torch.pow((rgb+0.055)/1.055, 2.4) * m1 + rgb/12.92*(1-m1)
    M1 = [[0.4124/0.95047, 0.3576/0.95047, 0.1805/0.95047], \
          [0.2126, 0.7152, 0.0722],\
          [0.0193/1.08883, 0.1192/1.08883, 0.9505/1.08883]]
    M1 = rgb.new(M1).view(1,3,3)
    xyz = torch.matmul(M1, rgb.view(bsz, c, h*w)).view(bsz, c, h, w)
    m2 = (xyz>0.008856).float()
    xyz = torch.pow(xyz, 1./3)*m2 + (xyz*7.787+16./116)*(1-m2)
    
    # Convert to Lab space
    l = 116. * xyz[:,1] - 16
    a = 500. * (xyz[:,0] - xyz[:,1])
    b = 200. * (xyz[:,1] - xyz[:,2])
    
    return torch.stack([l,a,b], dim=1)


def lab2rgb(img_in):
    '''
    transformation from Lab to RGB
    Input:
        img_in: (bsz, c, h, w) tensor of Lab image batch 
    Output:
        img_out: (bsz, c, h, w) tensof of rgb image batch with value range (-1, 1)
    '''
    
    # Convert to XYZ space
    y = (img_in[:,0] + 16.) / 116
    x = img_in[:, 1] / 500. + y
    z = y - img_in[:, 2] / 200.
    xyz = torch.stack([x, y, z], dim=1)
    xyz_pow_3 = xyz.pow(3)
    xyz = torch.where(xyz_pow_3 > 0.008856, xyz_pow_3, (xyz-16./116)/7.787)
    xyz = xyz * xyz.new([0.95047, 1.0, 1.08883]).view(1,3,1,1)
    
    # Convert to RGB space
    r = xyz[:,0]*  3.2406 + xyz[:,1]* -1.5372 + xyz[:,2]* -0.4986
    g = xyz[:,0]* -0.9689 + xyz[:,1]*  1.8758 + xyz[:,2]*  0.0415
    b = xyz[:,0]*  0.0557 + xyz[:,1]* -0.2040 + xyz[:,2]*  1.0570

    rgb = torch.stack([r,g,b], dim=1)
    rgb = torch.where(rgb>0.0031308, 1.055*rgb.pow(1./2.4)-0.055, rgb*12.92)
    
    return rgb
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms
from .utils import fspecial_gauss
from .SteerPyrComplex import SteerablePyramid
import math

    
class CW_SSIM(torch.nn.Module):
    '''
    This is an pytorch implementation of Complex-Wavelet 
    Structural SIMilarity (CW-SSIM) index. 

    M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey. 
    "Complex Wavelet Structural Similarity: A New Image Similarity Index", 
    IEEE Transactions on Image Processing, 18(11), 2385-401, 2009.

    Matlab version:
    https://www.mathworks.com/matlabcentral/fileexchange/43017-complex-wavelet-structural-similarity-index-cw-ssim
    '''
    def __init__(self, imgSize=[256,256], channels=3, level=4, ori=8, device = torch.device("cuda")):
        assert imgSize[0]==imgSize[1]
        super(CW_SSIM, self).__init__()
        self.ori = ori
        self.level = level
        self.channels = channels
        self.win7 = (torch.ones(channels,1,7,7)/(7*7)).to(device)
        s = imgSize[0]/2**(level-1)
        self.w = fspecial_gauss(s-7+1, s/4, 1).to(device)
        self.SP = SteerablePyramid(imgSize=imgSize, K=ori, N=level, hilb=True,device=device)

    def abs(self, x):
        return torch.sqrt(x[:,0,...]**2+x[:,1,...]**2+1e-12)

    def conj(self, x, y):
        a = x[:,0,...]
        b = x[:,1,...]
        c = y[:,0,...]
        d = -y[:,1,...]
        return torch.stack((a*c-b*d,b*c+a*d),dim=1)

    def conv2d_complex(self, x, win, groups = 1): 
        real = F.conv2d(x[:,0,...], win, groups = groups)# - F.conv2d(x[:,1], win, groups = groups)
        imaginary = F.conv2d(x[:,1,...], win, groups = groups)# + F.conv2d(x[:,0], win, groups = groups)
        return torch.stack((real,imaginary),dim=1)

    def cw_ssim(self, x, y):
        cw_x = self.SP(x)
        cw_y = self.SP(y)
        bandind = self.level
        band_cssim = []
        for i in range(self.ori):
            
            band1 = cw_x[bandind][:,:,:,i,:,:]
            band2 = cw_y[bandind][:,:,:,i,:,:]
            corr = self.conj(band1,band2)
            corr_band = self.conv2d_complex(corr, self.win7, groups = self.channels)
            varr = (self.abs(band1))**2+(self.abs(band2))**2
            varr_band = F.conv2d(varr, self.win7, stride=1, padding=0, groups = self.channels)
            cssim_map = (2*self.abs(corr_band) + 1e-12)/(varr_band + 1e-12)
            band_cssim.append((cssim_map*self.w.repeat(cssim_map.shape[0],1,1,1)).sum([2,3]).mean(1)) 

        return torch.stack(band_cssim,dim=1).mean(1)

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = self.cw_ssim(x, y)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = self.cw_ssim(x, y)
            return score
 
if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/r0.png')
    parser.add_argument('--dist', type=str, default='images/r1.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("L"),repeatNum=1).to(device)
    dist = prepare_image(Image.open(args.dist).convert("L"),repeatNum=1).to(device)
    dist.requires_grad_(True)

    model = CW_SSIM(imgSize=[256,256], channels=1, level=4, ori=8)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.9561

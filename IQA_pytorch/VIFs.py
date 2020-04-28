import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .utils import fspecial_gauss

class VIFs(nn.Module):
    def __init__(self, channels=3):
        super(VIFs, self).__init__()
        '''spatial domain VIF  
        https://live.ece.utexas.edu/research/Quality/VIF.htm
        '''
        self.channels = channels
        self.eps = 1e-10

    def vif(self, img1, img2):
        num = 0
        den = 0
        sigma_nsq=2
        channels = self.channels
        eps = self.eps
        for scale in range(1,5):
            N = 2**(4-scale+1)+1
            win = fspecial_gauss(N,N/5,channels).to(img1.device)
            
            if scale > 1:
                img1 = F.conv2d(img1, win, padding =0, groups = channels)
                img2 = F.conv2d(img2, win, padding =0, groups = channels)
                img1 = img1[:,:,0::2,0::2]
                img2 = img2[:,:,0::2,0::2]

            mu1 = F.conv2d(img1, win, padding =0, groups = channels)
            mu2 = F.conv2d(img2, win, padding =0, groups = channels)
            mu1_sq = mu1*mu1
            mu2_sq = mu2*mu2
            mu1_mu2 = mu1*mu2
            sigma1_sq = F.conv2d(img1*img1, win, padding =0, groups = channels) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, win, padding =0, groups = channels) - mu2_sq
            sigma12 = F.conv2d(img1*img2, win, padding =0, groups = channels) - mu1_mu2
            
            sigma1_sq = F.relu(sigma1_sq)
            sigma2_sq = F.relu(sigma2_sq)
            
            g = sigma12/(sigma1_sq+eps)
            sv_sq = sigma2_sq-g*sigma12
            sigma1_sq = F.relu(sigma1_sq-eps)

            g = g.masked_fill(sigma2_sq<eps,0)
            sv_sq = sv_sq.masked_fill(sigma2_sq<eps,0)
            
            sv_sq[g<0]=sigma2_sq[g<0]
            g = F.relu(g)
            sv_sq = sv_sq.masked_fill(sv_sq<eps,eps)
            
            x = g**2*sigma1_sq/(sv_sq+sigma_nsq) + 1
            y = sigma1_sq/sigma_nsq + 1
            num=num+torch.sum(torch.log10(x),dim=[1,2,3]) 
            den=den+torch.sum(torch.log10(y),dim=[1,2,3])

        return num/(den+1e-12)

    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = self.vif(x, y)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = self.vif(x, y)
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

    ref = prepare_image(Image.open(args.ref).convert("L")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("L")).to(device)

    model = VIFs(channels=1)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.2597


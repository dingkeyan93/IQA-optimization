import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms
from .utils import fspecial_gauss
from .SSIM import ssim

def ms_ssim(X, Y, win):
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    weights = torch.FloatTensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = ssim(X, Y, win=win, get_cs=True)
        mcs.append(cs)
        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0) 
    return msssim_val

class MS_SSIM(torch.nn.Module):
    def __init__(self, channels=3):
        super(MS_SSIM, self).__init__()
        self.win = fspecial_gauss(11, 1.5, channels)

    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if as_loss:
            score = ms_ssim(X, Y, win=self.win)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = ms_ssim(X, Y, win=self.win)
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

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)
    
    model = MS_SSIM(channels=3)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.8524

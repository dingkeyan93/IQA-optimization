import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms

class GMSD(nn.Module):
    # Refer to http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

    def __init__(self, channels=3):
        super(GMSD, self).__init__()
        self.channels = channels
        dx = (torch.Tensor([[1,0,-1],[1,0,-1],[1,0,-1]])/3.).unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1)
        dy = (torch.Tensor([[1,1,1],[0,0,0],[-1,-1,-1]])/3.).unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.dy = nn.Parameter(dy, requires_grad=False)
        self.aveKernel = nn.Parameter(torch.ones(channels,1,2,2)/4., requires_grad=False)

    def gmsd(self, img1, img2, T=170):
        Y1 = F.conv2d(img1, self.aveKernel, stride=2, padding =0, groups = self.channels)
        Y2 = F.conv2d(img2, self.aveKernel, stride=2, padding =0, groups = self.channels)

        IxY1 = F.conv2d(Y1, self.dx, stride=1, padding =1, groups = self.channels)
        IyY1 = F.conv2d(Y1, self.dy, stride=1, padding =1, groups = self.channels)
        gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2+1e-12)

        IxY2 = F.conv2d(Y2, self.dx, stride=1, padding =1, groups = self.channels)
        IyY2 = F.conv2d(Y2, self.dy, stride=1, padding =1, groups = self.channels)
        gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2+1e-12)
        
        quality_map = (2*gradientMap1*gradientMap2 + T)/(gradientMap1**2+gradientMap2**2 + T)
        score = torch.std(quality_map.view(quality_map.shape[0],-1),dim=1)
        return score
        
    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = self.gmsd(x, y)
            return score.mean()
        else:
            with torch.no_grad():
                score = self.gmsd(x, y)
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

    model = GMSD().to(device)

    score = model(ref, dist, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.1907


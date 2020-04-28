import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms
from .utils import fspecial_gauss
from .SteerPyrUtils import sp5_filters

def corrDn(image, filt, step=1, channels=1):

    filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1).to(image.device)
    p = (filt_.shape[2]-1)//2
    image = F.pad(image, (p,p,p,p),'reflect')
    img = F.conv2d(image, filt_, stride=step, padding=0, groups = channels)
    return img

def SteerablePyramidSpace(image, height=4, order=5, channels=1):
    num_orientations = order + 1
    filters = sp5_filters()
   
    hi0 = corrDn(image, filters['hi0filt'], step=1, channels=channels)
    pyr_coeffs = []
    pyr_coeffs.append(hi0)
    lo = corrDn(image, filters['lo0filt'], step=1, channels=channels)
    for _ in range(height):
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))
        for b in range(num_orientations):
            filt = filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
            band = corrDn(lo, filt, step=1, channels=channels)
            pyr_coeffs.append(band)
        lo = corrDn(lo, filters['lofilt'], step=2, channels=channels)

    pyr_coeffs.append(lo)
    return pyr_coeffs

 
if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/r0.png')
    parser.add_argument('--dist', type=str, default='images/r1.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dist = prepare_image(Image.open(args.dist).convert("L"),repeatNum=1).to(device)
    x = SteerablePyramidSpace(dist*255,channels=1)
    c = 0


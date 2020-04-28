import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.fft import ifftshift
import math
from .utils import abs, real, imag, spatial_normalize, downsample, prepare_image

eps = 1e-12

def logGabor(rows,cols,omega0,sigmaF):

    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)))
    mask = u1**2+u2**2<0.25

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1*mask)
    u2 = ifftshift(u2*mask)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2+eps)
    radius[0,0]=1

    logGaborDenom = 2. * sigmaF ** 2.
    logRadOverFo = (np.log(radius / omega0))
    log_Gabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
    log_Gabor[0, 0] = 0.  # Undo the radius fudge
    return log_Gabor

def rgb_to_lab_NCHW(srgb):
    srgb.clamp_(min=0,max=1)
    device = srgb.device
    B, C, rows, cols = srgb.shape
    srgb_pixels = torch.reshape(srgb.permute(0,2,3,1), [B, -1, 3])
    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor).to(device)#.unsqueeze(0).repeat(B,1,1)
	
    xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz)
	
    # XYZ to Lab
    t = torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).to(device)
    xyz_normalized_pixels = (xyz_pixels / t)

    epsilon = 6.0/29.0
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(device)
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(device)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels).clamp_(min=eps).pow(1/3)) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ]).type(torch.FloatTensor).to(device)#.unsqueeze(0).repeat(B,1,1)
    lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device)

    return torch.reshape(lab_pixels.permute(0,2,1), srgb.shape)

def SDSP(img,sigmaF,omega0,sigmaD,sigmaC):
    B, C, rows, cols = img.shape

    lab = rgb_to_lab_NCHW(img/255)
    LChannel, AChannel, BChannel = lab[:,0,:,:].unsqueeze(1),lab[:,1,:,:].unsqueeze(1),lab[:,2,:,:].unsqueeze(1)
    LFFT = torch.rfft(LChannel,2,onesided=False)
    AFFT = torch.rfft(AChannel,2,onesided=False)
    BFFT = torch.rfft(BChannel,2,onesided=False)
    
    LG = logGabor(rows,cols,omega0,sigmaF)
    LG = torch.from_numpy(LG).reshape(1, 1, rows,cols,1).repeat(B,1,1,1,2).float().to(img.device)

    FinalLResult = real(torch.ifft(LFFT*LG,2))
    FinalAResult = real(torch.ifft(AFFT*LG,2))
    FinalBResult = real(torch.ifft(BFFT*LG,2))

    SFMap = torch.sqrt(FinalLResult**2 + FinalAResult**2 + FinalBResult**2+eps)

    coordinateMtx = torch.from_numpy(np.arange(0,rows)).float().reshape(1,1,rows,1).repeat(B,1,1,cols).to(img.device)
    centerMtx = torch.ones_like(coordinateMtx)*rows/2
    coordinateMty = torch.from_numpy(np.arange(0,cols)).float().reshape(1,1,1,cols).repeat(B,1,rows,1).to(img.device)
    centerMty = torch.ones_like(coordinateMty)*cols/2
    SDMap = torch.exp(-((coordinateMtx - centerMtx)**2+(coordinateMty - centerMty)**2)/(sigmaD**2))

    normalizedA = spatial_normalize(AChannel)

    normalizedB = spatial_normalize(BChannel)

    labDistSquare = normalizedA**2 + normalizedB**2
    SCMap = 1 - torch.exp(-labDistSquare / (sigmaC**2))
    VSMap = SFMap * SDMap * SCMap

    normalizedVSMap = spatial_normalize(VSMap)
    return normalizedVSMap

def vsi(image1,image2):

    constForVS = 1.27
    constForGM = 386
    constForChrom = 130
    alpha = 0.40
    lamda = 0.020
    sigmaF = 1.34
    omega0 = 0.0210
    sigmaD = 145
    sigmaC = 0.001

    saliencyMap1 = SDSP(image1,sigmaF,omega0,sigmaD,sigmaC)
    saliencyMap2 = SDSP(image2,sigmaF,omega0,sigmaD,sigmaC)

    L1 = (0.06 * image1[:,0,:,:] + 0.63 * image1[:,1,:,:] + 0.27 * image1[:,2,:,:]).unsqueeze(1)
    L2 = (0.06 * image2[:,0,:,:] + 0.63 * image2[:,1,:,:] + 0.27 * image2[:,2,:,:]).unsqueeze(1)
    M1 = (0.30 * image1[:,0,:,:] + 0.04 * image1[:,1,:,:] - 0.35 * image1[:,2,:,:]).unsqueeze(1)
    M2 = (0.30 * image2[:,0,:,:] + 0.04 * image2[:,1,:,:] - 0.35 * image2[:,2,:,:]).unsqueeze(1)
    N1 = (0.34 * image1[:,0,:,:] - 0.60 * image1[:,1,:,:] + 0.17 * image1[:,2,:,:]).unsqueeze(1)
    N2 = (0.34 * image2[:,0,:,:] - 0.60 * image2[:,1,:,:] + 0.17 * image2[:,2,:,:]).unsqueeze(1)

    L1, L2 = downsample(L1, L2)
    M1, M2 = downsample(M1, M2)
    N1, N2 = downsample(N1, N2)
    saliencyMap1, saliencyMap2 = downsample(saliencyMap1, saliencyMap2)

    dx = torch.Tensor([[3, 0, -3], [10, 0, -10],  [3,  0, -3]]).float()/16
    dy = torch.Tensor([[3, 10, 3], [0, 0, 0],  [-3,  -10, -3]]).float()/16
    dx = dx.reshape(1,1,3,3).to(image1.device)
    dy = dy.reshape(1,1,3,3).to(image1.device)
    IxY1 = F.conv2d(L1, dx, stride=1, padding =1)     
    IyY1 = F.conv2d(L1, dy, stride=1, padding =1)    
    gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2+eps)
    IxY2 = F.conv2d(L2, dx, stride=1, padding =1)     
    IyY2 = F.conv2d(L2, dy, stride=1, padding =1)    
    gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2+eps)


    VSSimMatrix = (2 * saliencyMap1 * saliencyMap2 + constForVS) / (saliencyMap1**2 + saliencyMap2**2 + constForVS)
    gradientSimMatrix = (2*gradientMap1*gradientMap2 + constForGM) /(gradientMap1**2 + gradientMap2**2 + constForGM)

    weight = torch.max(saliencyMap1, saliencyMap2)

    ISimMatrix = (2 * M1 * M2 + constForChrom) / (M1**2 + M2**2 + constForChrom)
    QSimMatrix = (2 * N1 * N2 + constForChrom) / (N1**2 + N2**2 + constForChrom)

    # SimMatrixC = (torch.sign(gradientSimMatrix) * (torch.abs(gradientSimMatrix)+eps) ** alpha) * VSSimMatrix * \
    #     (torch.sign(ISimMatrix * QSimMatrix)*(torch.abs(ISimMatrix * QSimMatrix)+eps) ** lamda) * weight
    SimMatrixC = ((torch.abs(gradientSimMatrix)+eps) ** alpha) * VSSimMatrix * \
        ((torch.abs(ISimMatrix * QSimMatrix)+eps) ** lamda) * weight

    return torch.sum(SimMatrixC,dim=[1,2,3]) / torch.sum(weight,dim=[1,2,3])

class VSI(torch.nn.Module):
    # Refer to https://sse.tongji.edu.cn/linzhang/IQA/VSI/VSI.htm

    def __init__(self, channels=3):
        super(VSI, self).__init__()
        assert channels == 3

    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = vsi(x, y)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = vsi(x, y)
            return score


if __name__ == '__main__':
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/r0.png')
    parser.add_argument('--dist', type=str, default='images/r1.png')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)
    
    model = VSI().to(device)
    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.9322


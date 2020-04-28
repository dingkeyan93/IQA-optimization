import numpy as np
import os
import sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import inspect
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
from .utils import abs, real, imag, downsample

def lowpassfilter(size, cutoff, n):
    """
    Constructs a low-pass Butterworth filter:
        f = 1 / (1 + (w/cutoff)^2n)
    usage:  f = lowpassfilter(sze, cutoff, n)
    where:  size    is a tuple specifying the size of filter to construct
            [rows cols].
        cutoff  is the cutoff frequency of the filter 0 - 0.5
        n   is the order of the filter, the higher n is the sharper
            the transition is. (n must be an integer >= 1). Note
            that n is doubled so that it is always an even integer.
    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))

def filtergrid(rows, cols):

    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)

    return radius, u1, u2

def phasecong2(im):
    nscale          = 4
    norient         = 4
    minWaveLength   = 6
    mult            = 2
    sigmaOnf        = 0.55
    dThetaOnSigma   = 1.2
    k               = 2.0
    epsilon         = .0001
    thetaSigma = np.pi/norient/dThetaOnSigma

    _, _, rows,cols = im.shape
    imagefft = torch.rfft(im,2,onesided=False)
    
    lp = lowpassfilter((rows,cols),.45,15)

    radius, _, _ = filtergrid(rows, cols)
    radius[0, 0] = 1.
    logGaborList = []
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.
    for s in range(nscale):
        wavelength = minWaveLength * mult ** s
        fo = 1. / wavelength  # Centre frequency of filter
        logRadOverFo = (np.log(radius / fo))
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      # Apply the low-pass filter
        logGabor[0, 0] = 0.  # Undo the radius fudge
        logGaborList.append(logGabor)

    # Matrix of radii
    cy = np.floor(rows/2)
    cx = np.floor(cols/2)
    y, x = np.mgrid[0:rows, 0:cols]
    y = (y-cy)/rows
    x = (x-cx)/cols
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-y, x)
    radius = ifftshift(radius)      # Quadrant shift radius and theta so that filters
    theta  = ifftshift(theta)       # are constructed with 0 frequency at the corners.
    radius[0,0] = 1 
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    spreadList = []
    for o in np.arange(norient):
        angl = o*np.pi/norient # Filter angle.
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)      # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)      # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.
        # dtheta = np.minimum(dtheta*NumberAngles/2, math.pi)
        spread = np.exp((-dtheta**2) / (2 * thetaSigma**2));  # Calculate the angular
        spreadList.append(spread)

    ifftFilterArray = [[],[],[],[]]
    filterArray = [[],[],[],[]]
    for o in np.arange(norient):
        for s in np.arange(nscale):
            filter = logGaborList[s] * spreadList[o]
            filterArray[o].append(torch.from_numpy(filter).reshape(1,1,rows,cols).float().to(im.device))
            ifftFilt = np.real(ifft2(filter))*math.sqrt(rows*cols)
            ifftFilterArray[o].append(torch.from_numpy(ifftFilt).reshape(1,1,rows,cols).float().to(im.device))

    EnergyAll = 0
    AnAll = 0
    for o in np.arange(norient):
        sumE_ThisOrient   = 0  
        sumO_ThisOrient   = 0
        sumAn_ThisOrient  = 0
        Energy            = 0
        MatrixEOList = []
        for s in np.arange(nscale):
            filter = filterArray[o][s]
            c = imagefft * filter.unsqueeze(-1).repeat(1,1,1,1,2)
            MatrixEO = torch.ifft(imagefft * filter.unsqueeze(-1).repeat(1,1,1,1,2), 2)
            MatrixEOList.append(MatrixEO)
            
            An = abs(MatrixEO)    # Amplitude of even & odd filter response.
            sumAn_ThisOrient = sumAn_ThisOrient + An             # Sum of amplitude responses.
            sumE_ThisOrient = sumE_ThisOrient + real(MatrixEO) # Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + imag(MatrixEO) # Sum of odd filter convolution results.

            if s == 0:
                EM_n = torch.sum(filter**2,dim=[1,2,3])
                maxAn = An
            else:
                maxAn = torch.max(maxAn,An)  

        XEnergy = torch.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2+1e-12) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy
        for s in np.arange(nscale):
            EO = MatrixEOList[s]
            E = real(EO)
            O = imag(EO)
            Energy = Energy + E*MeanE + O*MeanO - torch.abs(E*MeanO - O*MeanE)

        meanE2n = torch.median((abs(MatrixEOList[0])**2).view(im.shape[0],-1),dim=1)[0] / -math.log(0.5)

        noisePower = meanE2n/EM_n
        EstSumAn2 = 0
        for s in np.arange(nscale):
            EstSumAn2 = EstSumAn2 + ifftFilterArray[o][s]**2
        EstSumAiAj = 0
        for si in np.arange(nscale-1):
            for sj in np.arange(si+1,nscale):
                EstSumAiAj = EstSumAiAj + ifftFilterArray[o][si]*ifftFilterArray[o][sj]
        sumEstSumAn2 = torch.sum(EstSumAn2,dim=[1,2,3])
        sumEstSumAiAj = torch.sum(EstSumAiAj,dim=[1,2,3])
        
        EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2 + 4*noisePower*sumEstSumAiAj
        
        tau = torch.sqrt(EstNoiseEnergy2/2+1e-12)                    
        EstNoiseEnergySigma = torch.sqrt( (2-math.pi/2)*tau**2 +1e-12)
        T =  tau*math.sqrt(math.pi/2) + k*EstNoiseEnergySigma
        T = T/1.7
        Energy = F.relu(Energy - T.view(-1,1,1,1))
        
        EnergyAll = EnergyAll + Energy
        AnAll = AnAll + sumAn_ThisOrient

    ResultPC = EnergyAll / AnAll
    return ResultPC

def fsim(imageRef, imageDis):
    
    channels = imageRef.shape[1]
    if channels == 3:
        Y1 = (0.299 * imageRef[:,0,:,:] + 0.587 * imageRef[:,1,:,:] + 0.114 * imageRef[:,2,:,:]).unsqueeze(1)
        Y2 = (0.299 * imageDis[:,0,:,:] + 0.587 * imageDis[:,1,:,:] + 0.114 * imageDis[:,2,:,:]).unsqueeze(1)
        I1 = (0.596 * imageRef[:,0,:,:] - 0.274 * imageRef[:,1,:,:] - 0.322 * imageRef[:,2,:,:]).unsqueeze(1)
        I2 = (0.596 * imageDis[:,0,:,:] - 0.274 * imageDis[:,1,:,:] - 0.322 * imageDis[:,2,:,:]).unsqueeze(1)
        Q1 = (0.211 * imageRef[:,0,:,:] - 0.523 * imageRef[:,1,:,:] + 0.312 * imageRef[:,2,:,:]).unsqueeze(1)
        Q2 = (0.211 * imageDis[:,0,:,:] - 0.523 * imageDis[:,1,:,:] + 0.312 * imageDis[:,2,:,:]).unsqueeze(1)
        Y1, Y2 = downsample(Y1, Y2)
        I1, I2 = downsample(I1, I2)
        Q1, Q2 = downsample(Q1, Q2)
    elif channels == 1:
        Y1, Y2 = downsample(imageRef, imageDis)
    else:
        raise ValueError('channels error')

    PC1 = phasecong2(Y1)
    PC2 = phasecong2(Y2)

    dx = torch.Tensor([[3, 0, -3], [10, 0, -10],  [3,  0, -3]]).float()/16
    dy = torch.Tensor([[3, 10, 3], [0, 0, 0],  [-3,  -10, -3]]).float()/16
    dx = dx.reshape(1,1,3,3).to(imageRef.device)
    dy = dy.reshape(1,1,3,3).to(imageRef.device)
    IxY1 = F.conv2d(Y1, dx, stride=1, padding =1)    
    IyY1 = F.conv2d(Y1, dy, stride=1, padding =1)    
    gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2+1e-12)
    IxY2 = F.conv2d(Y2, dx, stride=1, padding =1)  
    IyY2 = F.conv2d(Y2, dy, stride=1, padding =1)
    gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2+1e-12)

    T1 = 0.85  
    T2 = 160
    PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
    gradientSimMatrix = (2*gradientMap1*gradientMap2 + T2)/(gradientMap1**2 + gradientMap2**2 + T2)
    PCm = torch.max(PC1, PC2)
    SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
    FSIM_val = torch.sum(SimMatrix,dim=[1,2,3]) / torch.sum(PCm,dim=[1,2,3])
    if channels==1:
        return FSIM_val

    T3 = 200
    T4 = 200
    ISimMatrix = (2 * I1 * I2 + T3) / (I1**2 + I2**2 + T3)
    QSimMatrix = (2 * Q1 * Q2 + T4) / (Q1**2 + Q2**2 + T4)

    SimMatrixC = gradientSimMatrix * PCSimMatrix * PCm * \
        torch.sign(gradientSimMatrix) * ((torch.abs(ISimMatrix * QSimMatrix)+1e-12) ** 0.03)

    return  torch.sum(SimMatrixC,dim=[1,2,3]) / torch.sum(PCm,dim=[1,2,3]) 

class FSIM(torch.nn.Module):
    # Refer to https://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm
    
    def __init__(self, channels=3):
        super(FSIM, self).__init__()

    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = fsim(x, y)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = fsim(x, y)
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

    ref = prepare_image(Image.open(args.ref).convert("RGB"), repeatNum = 1).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB"), repeatNum = 1).to(device)

    model = FSIM(channels=3).to(device)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.7843


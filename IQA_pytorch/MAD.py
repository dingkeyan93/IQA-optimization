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
from .utils import abs, real, imag, downsample, batch_fftshift2d, batch_ifftshift2d

MAX = nn.MaxPool2d((2,2), stride=1, padding=1)

def extract_patches_2d(img, patch_shape=[64, 64], step=[27,27], batch_first=False, keep_last_patch=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0) and keep_last_patch:
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0) and keep_last_patch:
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches
    
def make_csf(rows, cols, nfreq=32):
    xvals = np.arange(-(cols - 1) / 2., (cols + 1) / 2.) 
    yvals = np.arange(-(rows - 1) / 2., (rows + 1) / 2.) 

    xplane,yplane=np.meshgrid(xvals, yvals)	# generate mesh
    plane=((xplane+1j *yplane)/cols)*2*nfreq
    radfreq=np.abs(plane)				# radial frequency

    w=0.7
    s=(1-w)/2*np.cos(4*np.angle(plane))+(1+w)/2
    radfreq=radfreq/s

    # Now generate the CSF
    csf = 2.6*(0.0192+0.114*radfreq)*np.exp(-(0.114*radfreq)**1.1)
    csf[radfreq < 7.8909]=0.9809

    return np.transpose(csf)

def get_moments(d,sk=False):
    # Return the first 4 moments of the data provided
    mean = torch.mean(d,dim=[3,4],keepdim=True)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0),dim=[3,4],keepdim=True)
    std = torch.pow(var+1e-12, 0.5)
    if sk:
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0),dim=[3,4],keepdim=True)
        kurtoses = torch.mean(torch.pow(zscores, 4.0),dim=[3,4],keepdim=True) - 3.0  # excess kurtosis, should be 0 for Gaussian
        return mean,std,skews,kurtoses
    else:
        return mean,std

def ical_stat(x, p=16,s=4):
    B, C, H, W = x.shape
    x1 = extract_patches_2d(x,patch_shape=[p,p],step=[s,s])
    _,std,skews,kurt = get_moments(x1,sk=True)
    STD = std.reshape(B, C, (H-(p-s))//s, (W-(p-s))//s)
    SKEWS = skews.reshape(B, C, (H-(p-s))//s, (W-(p-s))//s)
    KURT = kurt.reshape(B, C, (H-(p-s))//s, (W-(p-s))//s)
    return STD, SKEWS, KURT # different with original version

def ical_std(x, p=16,s=4):
    B, C, H, W = x.shape
    x1 = extract_patches_2d(x,patch_shape=[p,p],step=[s,s])
    mean,std = get_moments(x1)
    mean = mean.reshape(B, C, (H-(p-s))//s, (W-(p-s))//s)
    std = std.reshape(B, C, (H-(p-s))//s, (W-(p-s))//s)

    return mean, std

def min_std(x, p=8,s=4):
    B, C, H, W = x.shape
    STD = torch.zeros_like(x)
    for i in range(0,H-p+1,s):
        for j in range(0,W-p+1,s):
            x1 = x[:,:,i:i+p,j:j+p]
            STD[:,:,i:i+s,j:j+s]=torch.min(torch.min(x1,2,keepdim=True)[0],3,keepdim=True)[0].repeat(1,1,s,s)
    return STD

def hi_index(ref_img, dst_img):
    k = 0.02874
    G = 0.5        
    C_slope = 1    
    Ci_thrsh= -5   
    Cd_thrsh= -5   
    # ms_scale= 1    

    ref = k * (ref_img+1e-12) ** (2.2/3)
    dst = k * (torch.abs(dst_img)+1e-12) ** (2.2/3)
    
    B, C, H, W = ref.shape

    csf = make_csf(H, W, 32)
    csf = torch.from_numpy(csf.reshape(1,1,H,W,1)).float().repeat(1,C,1,1,2).to(ref.device)
    x = torch.rfft(ref, 2, onesided=False)
    x1 = batch_fftshift2d(x)
    x2 = batch_ifftshift2d( x1 * csf )
    ref = real(torch.ifft(x2,2))
    x = torch.rfft(dst, 2, onesided=False)
    x1 = batch_fftshift2d(x)
    x2 = batch_ifftshift2d( x1 * csf )
    dst = real(torch.ifft(x2,2))

    m1_1,std_1 = ical_std(ref)
    B, C, H1, W1 = m1_1.shape
    # _,std_1 = ical_std(ref,p=8)
    # std_11 = min_std(std_1)
    std_1 = (-MAX(-std_1)/2)[:,:,:H1,:W1]
    _,std_2 = ical_std(dst-ref)

    BSIZE = 16
    eps = 1e-12
    Ci_ref = torch.log(torch.abs((std_1+eps)/(m1_1+eps)) )
    Ci_dst = torch.log(torch.abs((std_2+eps)/(m1_1+eps)) )
    Ci_dst = Ci_dst.masked_fill(m1_1 < G,-1000)
    idx1      = (Ci_ref > Ci_thrsh) & (Ci_dst > (C_slope * (Ci_ref - Ci_thrsh) + Cd_thrsh) ) 
    idx2      = (Ci_ref <= Ci_thrsh) & (Ci_dst > Cd_thrsh) 

    msk = Ci_ref.clone()
    msk = msk.masked_fill(~idx1,0)
    msk = msk.masked_fill(~idx2,0)
    msk[idx1] = Ci_dst[idx1] - (C_slope * (Ci_ref[idx1]-Ci_thrsh) + Cd_thrsh)
    msk[idx2] = Ci_dst[idx2] - Cd_thrsh

    win = torch.ones( (1,1,BSIZE, BSIZE) ).repeat(C,1,1,1).to(ref.device) / BSIZE**2
    xx = (ref_img-dst_img)**2
    # p = (BSIZE-1)//2
    # xx = F.pad(xx,(p,p,p,p),'reflect')
    lmse  = F.conv2d(xx, win, stride=4, padding =0, groups = C) 

    mp = msk * lmse
    # mp2 = mp[:,:, BSIZE+1:-BSIZE-1, BSIZE+1:-BSIZE-1]
    B, C, H, W = mp.shape
    return torch.norm( mp.reshape(B,C,-1) , dim=2 ) / math.sqrt( H*W ) * 200

def gaborconvolve(im):

    nscale          = 5      #Number of wavelet scales.
    norient         = 4      #Number of filter orientations.
    minWaveLength   = 3      #Wavelength of smallest scale filter.
    mult            = 3      #Scaling factor between successive filters.
    sigmaOnf        = 0.55   #Ratio of the standard deviation of the
    wavelength      = [minWaveLength,minWaveLength*mult,minWaveLength*mult**2, minWaveLength*mult**3, minWaveLength*mult**4]
    dThetaOnSigma   = 1.5    #Ratio of angular interval between filter orientations

    B, C, rows, cols = im.shape
    imagefft    = torch.rfft(im,2, onesided=False)            # Fourier transform of image

    # Pre-compute to speed up filter construction
    x = np.ones((rows,1)) * np.arange(-cols/2.,(cols/2.))/(cols/2.)
    y = np.dot(np.expand_dims(np.arange(-rows/2.,(rows/2.)),1) , np.ones((1,cols))/(rows/2.))
    radius = np.sqrt(x**2 + y**2)       # Matrix values contain *normalised* radius from centre.
    radius[int(np.round(rows/2+1)),int(np.round(cols/2+1))] = 1 # Get rid of the 0 radius value in the middle
    radius = np.log(radius+1e-12)

    theta = np.arctan2(-y,x)              # Matrix values contain polar angle.
    # (note -ve y is used to give +ve
    # anti-clockwise angles)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    thetaSigma = math.pi/norient/dThetaOnSigma  # Calculate the standard deviation of the

    logGabors = []
    for s in range(nscale):                  # For each scale.
        # Construct the filter - first calculate the radial filter component.
        fo = 1.0/wavelength[s]                  # Centre frequency of filter.
        rfo = fo/0.5                         # Normalised radius from centre of frequency plane
        # corresponding to fo.
        tmp = -(2 * np.log(sigmaOnf)**2)
        tmp2= np.log(rfo)
        logGabors.append(np.exp( (radius-tmp2)**2 /tmp))
        logGabors[s][int(np.round(rows/2)), int(np.round(cols/2))]=0


    E0 = [[],[],[],[]]
    for o in range(norient):                   # For each orientation.
        angl = o*math.pi/norient           # Calculate filter angle.
        
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)     # Difference in sine.
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)     # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                           # Absolute angular distance.
        spread = np.exp((-dtheta**2) / (2 * thetaSigma**2))      # Calculate the angular filter component.
        
        for s in range(nscale):        # For each scale.

            filter = fftshift(logGabors[s] * spread)
            filter = torch.from_numpy(filter).reshape(1,1,rows,cols,1).repeat(1,C,1,1,2).to(im.device)
            # c =  imagefft * filter
            e0 = torch.ifft( imagefft * filter, 2 )
            E0[o].append(e0)

    return E0
        
def lo_index(ref, dst):
    gabRef  = gaborconvolve( ref )
    gabDst  = gaborconvolve( dst )
    s = [0.5/13.25, 0.75/13.25, 1/13.25, 5/13.25, 6/13.25]

    BSIZE = 16
    mp = 0
    for gb_i in range(4):
        for gb_j in range(5):
            stdref, skwref, krtref = ical_stat( abs( gabRef[gb_i][gb_j] ) )
            stddst, skwdst, krtdst  = ical_stat( abs( gabDst[gb_i][gb_j] ) )
            mp = mp + s[gb_i] * ( torch.abs( stdref - stddst ) + 2*torch.abs( skwref - skwdst ) +  torch.abs( krtref - krtdst ) ) 

    # mp2 = mp[:,:, BSIZE+1:-BSIZE-1, BSIZE+1:-BSIZE-1]
    B, C, rows, cols = mp.shape
    return torch.norm( mp.reshape(B,C,-1) , dim=2 ) / np.sqrt(rows * cols) 


def mad(ref, dst):
    HI = hi_index(ref, dst)
    LO = lo_index(ref, dst)  
    thresh1   = 2.55
    thresh2   = 3.35
    b1        = math.exp(-thresh1/thresh2)
    b2        = 1 / (math.log(10)*thresh2)
    sig       = 1 / ( 1 + b1*HI**b2 ) 
    MAD = LO**(1-sig) * HI**(sig)
    return  MAD.mean(1)

class MAD(torch.nn.Module):
    # Refer to http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23

    def __init__(self, channels=3):
        super(MAD, self).__init__()

    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = mad(x, y)
            return score.mean()
        else:
            with torch.no_grad():
                score = mad(x, y)
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

    ref = prepare_image(Image.open(args.ref).convert("L"), repeatNum = 1).to(device)
    dist = prepare_image(Image.open(args.dist).convert("L"), repeatNum = 1).to(device)

    model = MAD(channels=1).to(device)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f', score.item())
    # score: 168
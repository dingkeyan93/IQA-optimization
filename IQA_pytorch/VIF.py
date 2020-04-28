import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms
from .utils import fspecial_gauss
from .SteerPyrSpace import SteerablePyramidSpace
import math

  
class VIF(torch.nn.Module):
    # Refer to https://live.ece.utexas.edu/research/Quality/VIF.htm

    def __init__(self, channels=3, level=4, ori=6, device = torch.device("cuda")):

        super(VIF, self).__init__()
        self.ori = ori-1
        self.level = level
        self.channels = channels
        self.M=3
        self.subbands=[4, 7, 10, 13, 16, 19, 22, 25]
        self.sigma_nsq=0.4
        self.tol = 1e-12

    def corrDn(self, image, filt, step=1, channels=1,start=[0,0],end=[0,0]):

        filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1).to(image.device)
        p = (filt_.shape[2]-1)//2
        image = F.pad(image, (p,p,p,p),'reflect')
        img = F.conv2d(image, filt_, stride=1, padding=0, groups = channels)
        img = img[:,:,start[0]:end[0]:step,start[1]:end[1]:step]
        return img
        
    def vifsub_est_M(self, org, dist):
        
        g_all = []
        vv_all = []
        for i in range(len(self.subbands)):
            sub=self.subbands[i]-1
            y=org[sub]
            yn=dist[sub]

            lev=np.ceil((sub-1)/6)
            winsize=int(2**lev+1)
            win = np.ones((winsize,winsize))

            newsizeX=int(np.floor(y.shape[2]/self.M)*self.M)
            newsizeY=int(np.floor(y.shape[3]/self.M)*self.M)
            y=y[:,:,:newsizeX,:newsizeY]
            yn=yn[:,:,:newsizeX,:newsizeY]

            winstart=[int(1*np.floor(self.M/2)),int(1*np.floor(self.M/2))]
            winend=[int(y.shape[2]-np.ceil(self.M/2))+1,int(y.shape[3]-np.ceil(self.M/2))+1]

            mean_x = self.corrDn(y,win/(winsize**2),step=self.M, channels=self.channels,start=winstart,end=winend)
            mean_y = self.corrDn(yn,win/(winsize**2),step=self.M, channels=self.channels,start=winstart,end=winend)
            cov_xy = self.corrDn(y*yn, win, step=self.M, channels=self.channels,start=winstart,end=winend) - (winsize**2)*mean_x*mean_y
            ss_x = self.corrDn(y**2,win, step=self.M, channels=self.channels,start=winstart,end=winend) - (winsize**2)*mean_x**2
            ss_y = self.corrDn(yn**2,win, step=self.M, channels=self.channels,start=winstart,end=winend) - (winsize**2)*mean_y**2

            ss_x = F.relu(ss_x)
            ss_y = F.relu(ss_y)
        
            g = cov_xy/(ss_x+self.tol)
            vv = (ss_y - g*cov_xy)/(winsize**2)
            
            g = g.masked_fill(ss_x < self.tol,0)
            vv [ss_x < self.tol] = ss_y [ss_x < self.tol]
            ss_x = ss_x.masked_fill(ss_x < self.tol,0)
            
            g = g.masked_fill(ss_y < self.tol,0)
            vv = vv.masked_fill(ss_y < self.tol,0)
            
            vv[g<0]=ss_y[g<0]
            g = F.relu(g)
            
            vv = vv.masked_fill(vv < self.tol, self.tol)
            
            g_all.append(g)
            vv_all.append(vv)
        return g_all, vv_all
     
    def refparams_vecgsm(self, org):
        ssarr, l_arr, cu_arr = [], [], []
        for i in range(len(self.subbands)):
            sub=self.subbands[i]-1
            y=org[sub]
            M = self.M
            newsizeX=int(np.floor(y.shape[2]/M)*M)
            newsizeY=int(np.floor(y.shape[3]/M)*M)
            y=y[:,:,:newsizeX,:newsizeY]
            B,C,H,W = y.shape
                        
            temp=[]
            for j in range(M):
                for k in range(M):
                    temp.append(y[:,:,k:H-(M-k)+1, j:W-(M-j)+1].reshape(B,C,-1))
            temp = torch.stack(temp,dim=3)
            mcu = torch.mean(temp,dim=2).unsqueeze(2).repeat(1,1,temp.shape[2],1)
            cu=torch.matmul((temp-mcu).permute(0,1,3,2),temp-mcu)/temp.shape[2]

            temp=[]
            for j in range(M):
                for k in range(M):
                    temp.append(y[:,:,k:H+1:M, j:W+1:M].reshape(B,C,-1))
            temp = torch.stack(temp,dim=2)
            ss=torch.matmul(torch.pinverse(cu),temp)
            # ss = torch.matmul(torch.pinverse(cu),temp)
            ss=torch.sum(ss*temp,dim=2)/(M*M)
            ss=ss.reshape(B,C,H//M,W//M)    
            v,_ = torch.symeig(cu,eigenvectors=True)
            l_arr.append(v)
            ssarr.append(ss)
            cu_arr.append(cu)

        return ssarr, l_arr, cu_arr

    def vif(self, x, y):
        sp_x = SteerablePyramidSpace(x, height=self.level, order=self.ori, channels=self.channels)[::-1]
        sp_y = SteerablePyramidSpace(y, height=self.level, order=self.ori, channels=self.channels)[::-1]
        g_all, vv_all = self.vifsub_est_M(sp_y, sp_x)
        ss_arr, l_arr, cu_arr = self.refparams_vecgsm(sp_y)
        num, den = [], []

        for i in range(len(self.subbands)):
            sub=self.subbands[i]
            g=g_all[i]
            vv=vv_all[i]
            ss=ss_arr[i]
            lamda = l_arr[i]
            neigvals=lamda.shape[2]
            lev=np.ceil((sub-1)/6)
            winsize=2**lev+1 
            offset=(winsize-1)/2
            offset=int(np.ceil(offset/self.M))
            
            _,_,H,W = g.shape
            g=  g[:,:,offset:H-offset,offset:W-offset]
            vv=vv[:,:,offset:H-offset,offset:W-offset]
            ss=ss[:,:,offset:H-offset,offset:W-offset]

            temp1=0 
            temp2=0
            for j in range(neigvals):
                cc = lamda[:,:,j].unsqueeze(2).unsqueeze(3)
                temp1=temp1+torch.sum(torch.log2(1+g*g*ss*cc/(vv+self.sigma_nsq)),dim=[2,3])
                temp2=temp2+torch.sum(torch.log2(1+ss*cc/(self.sigma_nsq)),dim=[2,3])
            num.append(temp1.mean(1))
            den.append(temp2.mean(1))

        return torch.stack(num,dim=1).sum(1)/(torch.stack(den,dim=1).sum(1)+1e-12)

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

    ref = prepare_image(Image.open(args.ref).convert("L"),repeatNum=1).to(device)
    dist = prepare_image(Image.open(args.dist).convert("L"),repeatNum=1).to(device)
    dist.requires_grad_(True)
    model = VIF(channels=1)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.1804

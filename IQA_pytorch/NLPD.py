import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                             dtype=np.float32)

class NLPD(nn.Module):
    """
    Normalised lapalcian pyramid distance.
    Refer to https://www.cns.nyu.edu/pub/eero/laparra16a-preprint.pdf
    https://github.com/alexhepburn/nlpd-tensorflow
    """
    def __init__(self, channels=3, k=6, filt=None):
        super(NLPD, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (channels, 1, 1)),
                              (channels, 1, 5, 5))
        self.k = k
        self.channels = channels
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()
        self.pad_one = nn.ReflectionPad2d(1)
        self.pad_two = nn.ReflectionPad2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

    def DN_filters(self):
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.1011, 0],
                                    [0.1493, 0, 0.1460],
                                    [0, 0.1015, 0.]]*self.channels,
                                   (self.channels,	1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0757, 0],
                                    [0.1986, 0, 0.1846],
                                    [0, 0.0837, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0477, 0],
                                    [0.2138, 0, 0.2243],
                                    [0, 0.0467, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2503, 0, 0.2616],
                                    [0, 0, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2598, 0, 0.2552],
                                    [0, 0, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2215, 0, 0.0717],
                                    [0, 0, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList([nn.Parameter(x, requires_grad=False)
                                     for x in dn_filts])
        sigmas = nn.ParameterList([nn.Parameter(torch.Tensor(np.array(x)),
                                  requires_grad=False) for x in sigmas])
        return dn_filts, sigmas

    def pyramid(self, im):
        out = []
        J = im
        pyr = []
        for i in range(0, self.k):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=self.channels)
            I_up = self.upsample(I)
            I_up_conv = F.conv2d(self.pad_two(I_up), self.filt, stride=1,
                                 padding=0, groups=self.channels)
            if J.size() != I_up_conv.size():
                I_up_conv = F.interpolate(I_up_conv, [J.size(2), J.size(3)])
            out = J - I_up_conv
            out_conv = F.conv2d(self.pad_one(torch.abs(out)), self.dn_filts[i],
                         stride=1, groups=self.channels)
            out_norm = out / (self.sigmas[i]+out_conv)
            pyr.append(out_norm)
            J = I
        return pyr

    def nlpd(self, x1, x2):
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)           
        total = []
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        score = torch.stack(total,dim=1).mean(1)
        return score

    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            score = self.nlpd(x, y)
            return score.mean()
        else:
            with torch.no_grad():
                score = self.nlpd(x, y)
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
    
    model = NLPD(channels=1).to(device)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.4016
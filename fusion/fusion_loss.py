import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from skimage import color

class fusionloss(nn.Module):
    def __init__(self):
        super(fusionloss, self).__init__()

    def forward(self,ir,vi,f):
        vi_gard = Sobelxy()(vi.mean(dim=1, keepdim=True))
        ir_gard = Sobelxy()(ir.mean(dim=1, keepdim=True))
        f_grad = Sobelxy()(f.mean(dim=1, keepdim=True))
        max_grad = torch.max(vi_gard, ir_gard)
        max_init = torch.max(vi, ir)
        grad_loss = F.l1_loss(max_grad, f_grad)
        init_loss = F.l1_loss(max_init, f)
        return grad_loss, init_loss


def local_color_loss(vi,fused):
    # vector_vi = F.normalize(vi, p=1, dim=1)
    # vector_f = F.normalize(fused, p=1, dim=1)
    # return torch.mean(1 - torch.sum(vector_vi * vector_f, dim=1, keepdim=True))
    vector_vi = F.normalize(vi, p=2, dim=1)
    vector_f = F.normalize(fused, p=2, dim=1)
    return torch.mean(1 - torch.sum(vector_vi * vector_f, dim=1, keepdim=True))



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def hist_loss(seg_pred, input_1, input_2):
    '''
    1. seg_pred transform to [1,2,3,2,3,1,3...] x batchsize
    2. Get class 1,2,3 index
    3. Use index to get value of img1 and img2
    4. Get hist of img1 and img2
    input_1=enhanced_img, input_2=gt_img--->input_1=if_img, input_2=vis_img
    '''

    N, C, H, W = seg_pred.shape
    seg_pred = seg_pred.reshape(N, C, -1)
    seg_pred_cls = seg_pred.argmax(dim=1)
    input_1 = input_1.reshape(N, 3, -1)
    input_2 = input_2.reshape(N, 3, -1)
    soft_hist = SoftHistogram(bins=256, min=0, max=1, sigma=400)
    # soft_hist = HardHistogram(bins=256, min=0, max=1)
    loss = []
    for n in range(N):
        cls = seg_pred_cls[n]
        img1 = input_1[n]
        img2 = input_2[n]
        for c in range(C):
            cls_index = torch.nonzero(cls == c).squeeze()
            img1_index = img1[:, cls_index]
            img2_index = img2[:, cls_index]
            for i in range(img1.shape[0]):
                img1_hist = soft_hist(img1_index[i])
                img2_hist = soft_hist(img2_index[i])
                loss.append(F.l1_loss(img1_hist, img2_hist))
    loss = 256 * sum(loss) / (N*C*H*W)
    return loss

class HardHistogram(nn.Module):
    def __init__(self, bins, min, max):
        super(HardHistogram, self).__init__()
        self.bins = bins
        self.min_val = min
        self.max_val = max
        self.delta = float(max - min) / float(bins)
        
    def forward(self, x):
        x = x.view(-1)
        x = torch.clamp(x, self.min_val, self.max_val)
        bin_idx = ((x - self.min_val) / self.delta).long()
        bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)
        histogram = torch.zeros(self.bins, device=x.device)
        histogram.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float))
        return histogram

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers.cuda()

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  
    return window

def mssim(img1, img2, window_size=11):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2
    (_, channel, height, width) = img1.size()

    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2) 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def std(img,  window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma1

def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    return ssim.mean()
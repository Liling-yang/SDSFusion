import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
from PIL import Image
import numpy as np
from torch.autograd import Variable
import argparse
import datetime
import time
import math
import logging
import os.path as osp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from thop import profile
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

from fusion_module import MFSM
from hrseg.hrseg_model import create_hrnet
import dataloader
from fusion_loss import fusionloss,hist_loss,local_color_loss

parser = argparse.ArgumentParser(description='MFSM')
parser.add_argument('--reuse', '-R', type=str, default=None) # 以防训练中断，调用某个权重继续训练
parser.add_argument('--test_only', default=True, help='set this option to test the model')
parser.add_argument('--stage', type=str, default='stage1')
args = parser.parse_args()

def rgb_to_ycbcr(img):
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    ycbcr_img = torch.stack([y, cb, cr], dim=1)
    return ycbcr_img

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

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
    ret = ssim_map.mean(1)  # Take mean over channel dimension
    return ret

def std(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma1

def final_ssim(img_ir, img_vis, img_fuse):
    img_ir_ycbcr = rgb_to_ycbcr(img_ir)
    img_vis_ycbcr = rgb_to_ycbcr(img_vis)
    img_fuse_ycbcr = rgb_to_ycbcr(img_fuse)
    ssim_ir_y = mssim(img_ir_ycbcr[:, 0:1], img_fuse_ycbcr[:, 0:1])
    ssim_vi_y = mssim(img_vis_ycbcr[:, 0:1], img_fuse_ycbcr[:, 0:1])

    ssim_vi_cbcr = mssim(img_vis_ycbcr[:, 1:3], img_fuse_ycbcr[:, 1:3])

    std_ir_y = std(img_ir_ycbcr[:, 0:1], window_size=11)
    std_vi_y = std(img_vis_ycbcr[:, 0:1], window_size=11)

    zero = torch.zeros_like(std_ir_y)
    one = torch.ones_like(std_vi_y)
    map1 = torch.where((std_ir_y - std_vi_y) > 0, one, zero).mean(1, keepdim=True)
    map2 = torch.where((std_ir_y - std_vi_y) >= 0, zero, one).mean(1, keepdim=True)

    ssim_y = map1 * ssim_ir_y.unsqueeze(1) + map2 * ssim_vi_y.unsqueeze(1)

    ssim = (ssim_y.mean() + ssim_vi_cbcr.mean()) / 2
    ssim = torch.clamp(ssim, 0, 1)
    return ssim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def train_fusion(writer=None):
    modelpth = './train_model'
    os.makedirs(modelpth, mode=0o777, exist_ok=True)
    fusion_batch_size = 10
    n_workers = 4
    
    ds = dataloader.fusion_dataset_loader('train')
    dl = torch.utils.data.DataLoader(ds, batch_size=fusion_batch_size,shuffle=True,num_workers=n_workers,pin_memory=False)

    net = MFSM()
    net.apply(weights_init)
    if args.reuse is not None:
        load_path = './train_model'
        load_path = os.path.join(load_path,'fusion_epoch_177.pth')
        net.load_state_dict(torch.load(load_path))
        print('Load Pre-trained Fusion Model:{}!'.format(load_path))

    net = net.cuda()
    net.train()

    seg_model = create_hrnet().cuda()
    lr_start = 1.8*1e-4
    optim = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=0.0001)
    criteria_fusion = fusionloss()
    st = glob_st = time.time()
    epoch = 150
    dl.n_iter = len(dl)

    for epo in range(0, epoch):
        # lr_decay=0.75
        # lr_this_epo=lr_start*lr_decay**((epo/20)+1)
        if epo <= epoch / 2:
            lr_this_epo = lr_start
        else:
            lr_this_epo = lr_start*0.1
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_epo
        for it,(image_vis,image_ir) in enumerate(dl):
            net.train()
            image_vis = image_vis.cuda()
            image_ir = image_ir.cuda()
            image_vis = Variable(image_vis)
            image_ir = Variable(image_ir)

            seg_map, seg_ft = seg_model(image_vis)
            I_f = net(image_ir,image_vis,seg_ft)

            loss_color = hist_loss(seg_map, I_f, image_vis)
            loss_grad, loss_init = criteria_fusion(image_ir,image_vis,I_f)
            loss_ssim = 1-final_ssim(image_ir,image_vis,I_f)
            loss_cos = local_color_loss(image_vis,I_f)
            loss_fusion = 10*loss_grad + 150*loss_init + 100*loss_color + 30*loss_ssim + 100*loss_cos
            
            optim.zero_grad()
            loss_fusion.backward()

            for name, param in net.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. Grad function: {param.grad_fn}")
            
            optim.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = dl.n_iter * epo + it + 1
            
            # 将loss记录到Tensorboard
            if writer is not None:
                writer.add_scalar('loss_fusion', loss_fusion.item(), now_it)
                writer.add_scalar('loss_cos', loss_cos.item(), now_it)
                writer.add_scalar('loss_init', loss_init.item(), now_it)
                writer.add_scalar('loss_grad', loss_grad.item(), now_it)
                writer.add_scalar('loss_color', loss_color.item(), now_it)
                writer.add_scalar('loss_ssim', loss_ssim.item(), now_it)
            
            if (it + 1) % 50 == 0: 
                lr = optim.param_groups[0]['lr']
                eta = int((dl.n_iter * epoch - now_it)* (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(
                    ['step: {it}/{max_it}',
                            'loss_fusion:{loss_fusion:.4f}\n',
                            'loss_cos: {loss_cos:.4f}',
                            'loss_init: {loss_init:.4f}',
                            'loss_grad: {loss_grad:4f}',
                            'loss_color: {loss_color:4f}',
                            'loss_ssim: {loss_ssim:4f}',
                            'eta: {eta}',
                            'time: {time:.4f}',]).format(
                        it=now_it,max_it=dl.n_iter * epoch,lr=lr,
                        loss_fusion=loss_fusion,
                        loss_cos=loss_cos,
                        loss_init=loss_init,
                        loss_grad=loss_grad,
                        loss_color=loss_color,
                        loss_ssim=loss_ssim,
                        eta=eta,time=t_intv,)
                print(msg)
                st = ed

            # if (now_it + 1) % 200 == 0: 
        save_pth = os.path.join(modelpth, f'fusion_epoch_{epo+1}.pth')
        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        torch.save(state, save_pth)

def pad_image_to_multiple(image, multiple_of=8):
    """
    Pad the image so that its dimensions are divisible by `multiple_of`.
    """
    w, h = image.size
    new_w = ((w - 1) // multiple_of + 1) * multiple_of
    new_h = ((h - 1) // multiple_of + 1) * multiple_of

    if new_w == w and new_h == h:
        return image

    padded_image = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded_image.paste(image, (0, 0))
    return padded_image

def remove_padding(image, original_size):
    """
    Crop the padded image back to the original size.
    """
    return image.crop((0, 0, original_size[0], original_size[1]))

def fusion_test():
    total_time = 0
    if os.path.basename(os.getcwd()) == 'SDSFusion':
        fusion_model_path = os.path.join(os.getcwd(),'pretrain/fusion/fusion_epoch_150.pth')
        fusion_dir = os.path.join(os.getcwd(), 'datasets/test/LLVIP/')
    elif os.path.basename(os.getcwd()) == 'fusion':
        fusion_model_path = os.path.join(os.getcwd(),'pretrain/fusion/fusion_epoch_150.pth')
        path1 = os.path.dirname(os.getcwd())
        fusion_dir = os.path.join(path1, 'datasets/test/LLVIP/')
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
    
    fusionmodel = MFSM().cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path), strict=False)
    
    segmodel = create_hrnet().cuda()
    segmodel.eval()
    
    testdataset = dataloader.fusion_dataset_loader_test(fusion_dir)
    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    
    testtqdm = tqdm(testloader, total=len(testloader))

    with torch.no_grad():
        for images_vis, images_ir, name in testtqdm:
            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()
            start_time = time.time()

            # Prepare input image for model
            original_size = (images_vis.shape[3], images_vis.shape[2])
            padded_images_vis = pad_image_to_multiple(transforms.ToPILImage()(images_vis.squeeze(0).cpu()), multiple_of=8)
            padded_images_ir = pad_image_to_multiple(transforms.ToPILImage()(images_ir.squeeze(0).cpu()), multiple_of=8)

            vis_tensor = transforms.ToTensor()(padded_images_vis).unsqueeze(0).cuda()
            ir_tensor = transforms.ToTensor()(padded_images_ir).unsqueeze(0).cuda()

            seg_map, seg_ft = segmodel(vis_tensor)
            I_f = fusionmodel(ir_tensor, vis_tensor, seg_ft)

            # Convert result to PIL image
            image_I_f = I_f.squeeze()
            image_I_f = (image_I_f * 255).byte()
            image_I_f = transforms.ToPILImage()(image_I_f.cpu())

            padded_result = pad_image_to_multiple(image_I_f, multiple_of=8)
            result_image = remove_padding(padded_result, original_size)

            if args.stage == 'stage1':
                save_path = os.path.join(fusion_dir, 'If-s1', name[0])
            if args.stage == 'stage2':
                save_path = os.path.join(fusion_dir, 'If-s2', name[0])

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result_image.save(save_path)

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

    average_time = total_time / len(testloader)
    print("Average time: %.3f s" % average_time)


if __name__ == "__main__":
    logpath='./logs'
    if args.test_only is False:
        writer = SummaryWriter('./logs')
        try:
            train_fusion(writer)
            print("Training Done!")
        finally:
            writer.close()
    else:
        fusion_test()
        print("Test Done!")
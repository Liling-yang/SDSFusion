import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
import math
import time
import datetime
from functools import reduce
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.misc as misc

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

        
class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.log_ag = torch.Tensor()  # 初始化 log_ag
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        path1 =os.getcwd()

        if args.load == '.':
            if os.path.basename(path1) == 'SDSFusion':
                self.dir = os.path.join(path1,'enhance_stage2','experiment',args.save)
            elif os.path.basename(path1) == 'enhance_stage2':
                self.dir = os.path.join(path1,'experiment',args.save)
        else:
            if os.path.basename(path1) == 'SDSFusion':
                self.dir = os.path.join(path1,'enhance_stage2','experiment',args.load)
            elif os.path.basename(path1) == 'enhance_stage2':
                self.dir = os.path.join(path1,'experiment',args.load)

            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                self.log_ag = torch.load(self.dir + '/ag_log.pt')  # 加载 log_ag
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, '', epoch, is_best=is_best)
        trainer.recompose.save(self.dir, 'R_', epoch, is_best=is_best)
        trainer.dis.save(self.dir, 'D_', epoch, is_best=is_best)
        trainer.loss.save(self.dir)

        # Save PSNR and AG logs
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(self.log_ag, os.path.join(self.dir, 'ag_log.pt'))  # 保存 log_ag
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def add_log_ag(self, log_ag):
        self.log_ag = torch.cat([self.log_ag, log_ag])  # 新增方法用于追加 log_ag

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale, epoch):
        filename = '{}/results/Epoch{}_{}_x{}_'.format( self.dir, epoch, filename, scale)
        postfix = ('HR', 'LR', 'HR_Pred', 'LR_NF_Pred', 'HR_Pred2', 'HR_b1', 'HR_b2', 'HR_b3', 'Mask1', 'Mask2', 'Mask3')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.jpg'.format(filename, p), ndarr)

    def save_unpaired_results(self, filename, save_list, original_sizes):
        if os.path.basename(os.getcwd()) == 'SDSFusion':
            filename1 = os.path.join(os.getcwd(),'datasets/test/LLVIP/vi_en-s2/{}'.format(filename))
        elif os.path.basename(os.getcwd()) == 'enhance_stage2':
            path1 = os.path.dirname(os.getcwd())
            filename1 = os.path.join(path1,'datasets/test/LLVIP/vi_en-s2/{}'.format(filename))
            
        postfix = ('LR', 'HR_Pred', 'LR_NF_Pred', 'HR_Pred2', 'HR_b1', 'HR_b2', 'HR_b3', 'Mask1', 'Mask2', 'Mask3')
        for v, p in zip(save_list, postfix):
            cropped = v[:, :, :original_sizes[0], :original_sizes[1]]
            normalized = cropped[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            if p == 'HR_Pred2':
                misc.imsave('{}.jpg'.format(filename1), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    # return -10 * math.log10(mse)
    psnr = -10 * math.log10(mse)

    # 计算 AG (清晰度)
    sr_gray = sr.mean(dim=1, keepdim=True)  # 将 SR 图像转换为灰度图
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).cuda()
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).cuda()

    grad_x = F.conv2d(sr_gray, sobel_x, padding=1)
    grad_y = F.conv2d(sr_gray, sobel_y, padding=1)

    grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    ag = grad_magnitude.mean().item()

    return psnr, ag

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

def apply_clahe_to_batch(tensor, clipLimit=1):
    b, c, h, w = tensor.shape
    result_tensor = torch.empty_like(tensor)
    for i in range(b):
        tensor_np = (tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imgr = tensor_np[:, :, 0]
        imgg = tensor_np[:, :, 1]
        imgb = tensor_np[:, :, 2]
        claher = cv2.createCLAHE(clipLimit, tileGridSize=(10, 18))
        claheg = cv2.createCLAHE(clipLimit, tileGridSize=(10, 18))
        claheb = cv2.createCLAHE(clipLimit, tileGridSize=(10, 18))
        cllr = claher.apply(imgr)
        cllg = claheg.apply(imgg)
        cllb = claheb.apply(imgb)
        rgb_img = np.stack((cllr, cllg, cllb), axis=-1)
        random_int = random.randint(0, 100)
        if random_int % 2 == 0:
            enhanced_img = rgb_img
        else:
            enhanced_img = tensor_np
        result_tensor[i] = torch.from_numpy(enhanced_img).permute(2, 0, 1).float() / 255.0
    return result_tensor


def apply_detail_enhance_to_batch(tensor, sigma_s=10, sigma_r=0.1):
    b, c, h, w = tensor.shape
    result_tensor = torch.empty_like(tensor)
    
    for i in range(b):
        tensor_np = (tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        enhanced_img = cv2.detailEnhance(tensor_np, sigma_s=sigma_s, sigma_r=sigma_r)
        result_tensor[i] = torch.from_numpy(enhanced_img).permute(2, 0, 1).float() / 255.0
    
    return result_tensor
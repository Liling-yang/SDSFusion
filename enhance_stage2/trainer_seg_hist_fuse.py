import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
import math
from decimal import Decimal

import utility
import IPython
import torch.nn.functional as F
import torch.nn as nn

import torchvision
import torch
from torch.autograd import Variable
from tqdm import tqdm
import pytorch_ssim
import torchvision
from PIL import Image
import numpy as np

from hrseg.hrseg_model import create_hrnet
import loss

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

class vgg_v2(nn.Module):
    def __init__(self, vgg_model):
        super(vgg_v2, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

def vgg_loss(vgg, img, gt):
    mse = nn.MSELoss(reduction='mean')
    img_vgg = vgg(img)
    gt_vgg = vgg(gt)

    return mse(img_vgg[0], gt_vgg[0]) + 0.6*mse(img_vgg[1], gt_vgg[1]) + 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])

def vgg_init(vgg_loc):
    vgg_model = torchvision.models.vgg16(pretrained=False).cuda()
    vgg_model.load_state_dict(torch.load(vgg_loc))
    trainable(vgg_model, False)

    return vgg_model

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

class Trainer():
    def __init__(self, args, loader, my_model, my_recompose, my_dis, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.recompose = my_recompose
        self.dis = my_dis
        self.loss = my_loss

        args.lr = 1e-4
        self.optimizer = utility.make_optimizer(args, self.recompose)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        args.lr = 1e-6
        self.optimizer_dis = utility.make_optimizer(args, self.dis)
        self.scheduler_dis = utility.make_scheduler(args, self.optimizer_dis)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.writer = SummaryWriter(log_dir=os.path.join(self.ckp.dir, 'logs'))
        if not hasattr(self.ckp, 'log_ag'):
            self.ckp.log_ag = torch.zeros_like(self.ckp.log)

    def train(self):
        self.scheduler.step()
        self.loss.step()

        epoch = self.scheduler.last_epoch + 1
        lr = 1e-4 #self.scheduler.get_lr()[0]，args.lr前面也要改！
        lr_dis = 1e-6 #self.scheduler_dis.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        criterion_ssim = pytorch_ssim.SSIM(window_size=11)
        criterion_mse = nn.MSELoss()

        adversarial_criterion = nn.MSELoss()

        if os.path.basename(os.getcwd()) == 'SDSFusion':
            path = os.path.join(os.getcwd(),'pretrain/vgg16-397923af.pth')
        elif os.path.basename(os.getcwd()) == 'enhance_stage2':
            path1 = os.path.dirname(os.getcwd())
            path = os.path.join(path1,'pretrain/vgg16-397923af.pth')
        vgg_model = vgg_init(path)
        # vgg_model = vgg_init('./pretrain/vgg16-397923af.pth')
        vgg = vgg_v2(vgg_model)
        vgg.eval()
        seg_model = create_hrnet().cuda()
        seg_model.eval()

        for batch, (lr, hr, lrr, hq, _, idx_scale) in enumerate(self.loader_train):
            lr, hr, lrr, hq = self.prepare(lr, hr, lrr, hq)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            lr = lr/255.0
            hr = hr/255.0
            hq = hq/255.0
            lrr = lrr/255.0
            [b, c, h, w] = hr.shape
            # phr1, phr2, phr3 = self.model(lr, 3)
            hr = utility.apply_detail_enhance_to_batch(hr, sigma_s=10, sigma_r=0.1)

            res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)

            'use seg_model'
            seg_map, seg_orin, seg_fea = seg_model(res_g3_s1)

            phr1, phr2, phr3 = self.model.forward_2(lr, res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2,
                                                    feat_g3_s4, seg_orin, seg_fea)

            Img_up = nn.Upsample(scale_factor=2, mode='bilinear')
            Img_up_4x = nn.Upsample(scale_factor=4, mode='bilinear')

            phr1_2 = Img_up_4x(phr3)
            phr2_2 = Img_up(phr2)
            phr3_2 = phr1

            # phr1_r, phr2_r, phr3_r = self.model(lrr, 3)
            res_g3_s11, res_g3_s21, res_g3_s41, feat_g3_s11, feat_g3_s21, feat_g3_s41 = self.model.forward_1(lrr, 3)

            'use seg_model'
            seg_map_, seg_orin_, seg_fea_ = seg_model(res_g3_s11)

            phr1_r, phr2_r, phr3_r = self.model.forward_2(lrr, res_g3_s11, res_g3_s21, res_g3_s41, feat_g3_s11, feat_g3_s21,
                                                    feat_g3_s41, seg_orin_, seg_fea_)

            phr1_2_r = Img_up_4x(phr3_r)
            phr2_2_r = Img_up(phr2_r)
            phr3_2_r = phr1_r

            input_step2 = [lr, phr1_2, phr2_2, phr3_2]
            input_step2_r = [lrr, phr1_2_r, phr2_2_r, phr3_2_r]

            phr, weight1, weight2, weight3 = self.recompose(input_step2, 3)
            phr_r, weight1_r, weight2_r, weight3_r = self.recompose(input_step2_r, 3)

            target_real = (torch.rand(self.args.batch_size*2, 1)*0.5 + 0.7).cuda()
            target_fake = (torch.rand(self.args.batch_size, 1)*0.3).cuda()
            ones_const = torch.ones(self.args.batch_size, 1).cuda()
            # ones_const = 0.8 + (0.95 - 0.8) * ones_const # soft-label

            self.optimizer_dis.zero_grad()

            hr_all = torch.cat((hr, hq), 0)

            hr_all_sorted, _ = hr_all.sort(dim=1)
            phr_r_sorted, _ = phr_r.sort(dim=1)

            hr_all_full = torch.cat((hr_all, hr_all_sorted), 1)
            phr_r_full = torch.cat((phr_r, phr_r_sorted), 1)

            discriminator_loss = adversarial_criterion(self.dis(hr_all_full, 3), target_real) + adversarial_criterion(self.dis(phr_r_full, 3), target_fake)
            discriminator_loss.backward(retain_graph=True)
            self.optimizer_dis.step()

            self.optimizer.zero_grad()
            ssim_loss = criterion_ssim(phr, hr)
            vi_gard = loss.Sobelxy()(phr.mean(dim=1, keepdim=True))
            high_grad = loss.Sobelxy()(hr.mean(dim=1, keepdim=True))
            grad_loss = F.l1_loss(vi_gard, high_grad)
            wssim_loss = criterion_ssim(weight1, weight1_r) + criterion_ssim(weight2, weight2_r) + criterion_ssim(weight3, weight3_r)
            generator_adversarial_loss = adversarial_criterion(self.dis(phr_r_full, 3), ones_const)
            full_loss = 1*ssim_loss + 1*wssim_loss + 1*grad_loss + 1*generator_adversarial_loss

            full_loss.backward()
            self.optimizer.step()

            if (batch + 1) % 50 == 0:
                self.ckp.write_log('[{}/{}]: Dis={:.4f}, SSIM={:.4f}, WSSIM={:.4f}, Grad={:.4f}, Gan={:.4f}, {:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    discriminator_loss.item(),
                    ssim_loss.item(),
                    wssim_loss.item(),
                    grad_loss.item(),
                    generator_adversarial_loss.item(),
                    timer_model.release(),
                    timer_data.release()))

                # Log losses to TensorBoard
                self.writer.add_scalar('Dis_Loss', discriminator_loss.item(), epoch * len(self.loader_train) + batch)
                self.writer.add_scalar('SSIM_Loss', ssim_loss.item(), epoch * len(self.loader_train) + batch)
                self.writer.add_scalar('WSSIM_Loss', wssim_loss.item(), epoch * len(self.loader_train) + batch)
                self.writer.add_scalar('Grad_Loss', grad_loss.item(), epoch * len(self.loader_train) + batch)
                self.writer.add_scalar('Gen_Loss', generator_adversarial_loss.item(), epoch * len(self.loader_train) + batch)
                self.writer.add_scalar('Full_Loss', full_loss.item(), epoch * len(self.loader_train) + batch)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.ckp.add_log_ag(torch.zeros(1, len(self.scale)))
        self.model.eval()
        seg_model = create_hrnet().cuda()
        seg_model.eval()
        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_psnr = 0
                eval_ag = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for idx_img, (lr, hr, lrr, hq, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr, lrr, hq = self.prepare(lr, hr, lrr, hq)
                    else:
                        lr, = self.prepare(lr)

                    lr = lr/255.0
                    hr = hr/255.0
                    hq = hq/255.0
                    lrr = lrr/255.0

                    [b, c, h, w] = hr.shape

                    # phr1, phr2, phr3 = self.model(lr, 3)
                    res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)
                    seg_map, seg_orin, seg_fea = seg_model(res_g3_s1)
                    phr1, phr2, phr3 = self.model.forward_2(lr, res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2,
                                                            feat_g3_s4, seg_orin, seg_fea)

                    Img_up = nn.Upsample(scale_factor=2, mode='bilinear')
                    Img_up_4x = nn.Upsample(scale_factor=4, mode='bilinear')

                    phr1_2 = Img_up_4x(phr3)
                    phr2_2 = Img_up(phr2)
                    phr3_2 = phr1

                    input_step2 = [lr, phr1_2, phr2_2, phr3_2]
                    phr, m1, m2, m3 = self.recompose(input_step2, 3)

                    phr3 = utility.quantize(phr3_2*255, self.args.rgb_range)
                    lr = utility.quantize(lr*255, self.args.rgb_range)
                    hr = utility.quantize(hr*255, self.args.rgb_range)
                    plr_nf = utility.quantize(lr*255, self.args.rgb_range)
                    phr = utility.quantize(phr*255, self.args.rgb_range)

                    phr1_2 = utility.quantize(phr1_2*255, self.args.rgb_range)
                    phr2_2 = utility.quantize(phr2_2*255, self.args.rgb_range)
                    phr3_2 = utility.quantize(phr3_2*255, self.args.rgb_range)

                    m1 = utility.quantize(m1/2*255, self.args.rgb_range)
                    m2 = utility.quantize(m2/2*255, self.args.rgb_range)
                    m3 = utility.quantize(m3/2*255, self.args.rgb_range)

                    save_list = [hr, lr, phr3, plr_nf, phr, phr1_2, phr2_2, phr3_2, m1, m2, m3]
                
                    if not no_eval:
                        psnr, ag = utility.calc_psnr(
                            phr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark)
                        eval_psnr += psnr
                        eval_ag += ag

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.log_ag[-1, idx_scale] = eval_ag / len(self.loader_test)
                best_psnr = self.ckp.log.max(0)
                best_ag = self.ckp.log_ag.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\tAG: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best_psnr[0][idx_scale],
                        best_psnr[1][idx_scale] + 1,
                        self.ckp.log_ag[-1, idx_scale],
                        best_ag[0][idx_scale],
                        best_ag[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_psnr[1][0] + 1 == epoch)) # .best is standard for psnr

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

    def close_writer(self):
        self.writer.close()

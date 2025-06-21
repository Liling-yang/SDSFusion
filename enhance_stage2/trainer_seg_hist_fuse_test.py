import os
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
from fvcore.nn import FlopCountAnalysis

def pad_to_divisible_by_8(tensor):
    _, _, h, w = tensor.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    padding = (0, pad_w, 0, pad_h)
    return F.pad(tensor, padding, mode='constant', value=0), (h, w)

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
    mse = nn.MSELoss(size_average=True)
    img_vgg = vgg(img)
    gt_vgg = vgg(gt)

    #return 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])
    return mse(img_vgg[0], gt_vgg[0]) + 0.6*mse(img_vgg[1], gt_vgg[1]) + 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])

def vgg_init(vgg_loc):
    vgg_model = torchvision.models.vgg16(pretrained = False).cuda()
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
        # self.loader_test = loader.loader_test
        self.loader_test = loader.loader_test_unpaired

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

    def train(self):
        self.scheduler.step()
        self.loss.step()

        epoch = self.scheduler.last_epoch + 1
        lr =1e-4 #self.scheduler.get_lr()[0]
        lr_dis = 1e-6 #self.scheduler_dis.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)
        criterion_mse = nn.MSELoss()

        adversarial_criterion = nn.MSELoss()

        if os.path.basename(os.getcwd()) == 'SDSFusion':
            path = torch.load(os.path.join(os.getcwd(),'pretrain/vgg16-397923af.pth'))
        elif os.path.basename(os.getcwd()) == 'enhance_stage2':
            path1 = os.path.dirname(os.getcwd())
            path = torch.load(os.path.join(path1,'pretrain/vgg16-397923af.pth'))
        vgg_model = vgg_init(path)

        vgg = vgg_v2(vgg_model)
        vgg.eval()

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

            phr1, phr2, phr3 = self.model(lr, 3)

            Img_up = nn.Upsample(scale_factor=2, mode='bilinear')
            Img_up_4x = nn.Upsample(scale_factor=4, mode='bilinear')

            phr1_2 = Img_up_4x(phr3)
            phr2_2 = Img_up(phr2)
            phr3_2 = phr1

            phr1_r, phr2_r, phr3_r = self.model(lrr, 3)
            phr1_2_r = Img_up_4x(phr3_r)
            phr2_2_r = Img_up(phr2_r)
            phr3_2_r = phr1_r


            input_step2 = [lr, phr1_2, phr2_2, phr3_2]
            input_step2_r = [lrr, phr1_2_r, phr2_2_r, phr3_2_r]

            phr, _, _, _ = self.recompose(input_step2, 3)
            phr_r, _, _, _ = self.recompose(input_step2_r, 3)

            target_real = (torch.rand(self.args.batch_size*2, 1)*0.5 + 0.7).cuda()
            target_fake = (torch.rand(self.args.batch_size, 1)*0.3).cuda()
            ones_const = torch.ones(self.args.batch_size, 1).cuda()

            self.optimizer_dis.zero_grad()

            hr_all = torch.cat((hr, hq), 0)
            phr_all = torch.cat((phr, phr_r), 0)

            discriminator_loss = adversarial_criterion(self.dis(hr_all, 3), target_real) + adversarial_criterion(self.dis(phr_r, 3), target_fake)
            discriminator_loss.backward(retain_graph=True)
            self.optimizer_dis.step()

            self.optimizer.zero_grad()
            rect_loss =  vgg_loss(vgg, phr, hr) + criterion_ssim(phr, hr) #+ 0.1*vgg_loss(vgg, phr_r, phr3_r.detach()) + criterion_ssim(phr_r, phr3_r)

            generator_adversarial_loss = adversarial_criterion(self.dis(phr, 3), ones_const)
            full_loss = rect_loss + 0.5*generator_adversarial_loss
            full_loss.backward()
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    discriminator_loss.item(),
                    rect_loss.item(),
                    generator_adversarial_loss.item(),
                    timer_model.release(),
                    timer_data.release()))

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        seg_model = create_hrnet().cuda()
        seg_model.eval()
        timer_test = utility.timer()

        printed_flops = False

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for idx_img, (lr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    lr, = self.prepare(lr)
                    lr, original_size = pad_to_divisible_by_8(lr)
                    lr = lr / 255.0
                    [b, c, h, w] = lr.shape

                    # Forward through model
                    res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)
                    seg_map, seg_orin, seg_fea = seg_model(res_g3_s1)
                    phr1, phr2, phr3 = self.model.forward_2(
                        lr, res_g3_s1, res_g3_s2, res_g3_s4,
                        feat_g3_s1, feat_g3_s2, feat_g3_s4,
                        seg_orin, seg_fea
                    )

                    Img_up = nn.Upsample(scale_factor=2, mode='bilinear')
                    Img_up_4x = nn.Upsample(scale_factor=4, mode='bilinear')
                    phr1_2 = Img_up_4x(phr3)
                    phr2_2 = Img_up(phr2)
                    phr3_2 = phr1

                    input_step2 = [lr, phr1_2, phr2_2, phr3_2]
                    phr, m1, m2, m3 = self.recompose(input_step2, 3)

                    phr3 = utility.quantize(phr3_2 * 255, self.args.rgb_range)
                    lr = utility.quantize(lr * 255, self.args.rgb_range)
                    plr_nf = utility.quantize(lr * 255, self.args.rgb_range)
                    phr = utility.quantize(phr * 255, self.args.rgb_range)

                    phr1_2 = utility.quantize(phr1_2 * 255, self.args.rgb_range)
                    phr2_2 = utility.quantize(phr2_2 * 255, self.args.rgb_range)
                    phr3_2 = utility.quantize(phr3_2 * 255, self.args.rgb_range)

                    m1 = utility.quantize(m1 / 2 * 255, self.args.rgb_range)
                    m2 = utility.quantize(m2 / 2 * 255, self.args.rgb_range)
                    m3 = utility.quantize(m3 / 2 * 255, self.args.rgb_range)

                    save_list = [lr, phr3, plr_nf, phr, phr1_2, phr2_2, phr3_2, m1, m2, m3]
                    self.ckp.save_unpaired_results(filename, save_list, original_size)

        self.ckp.write_log('Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        print('Average time: {:.3f}s\n'.format(timer_test.toc()/80))

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
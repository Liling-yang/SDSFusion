import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer_seg_hist_fuse_test import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)

    if os.path.basename(os.getcwd()) == 'SDSFusion':
        path = os.path.join(os.getcwd(),'pretrain/s2')
    elif os.path.basename(os.getcwd()) == 'enhance_stage2':
        path1 = os.path.dirname(os.getcwd())
        path = os.path.join(path1,'pretrain/s2')

    my_model = model.Model(args, checkpoint)
    my_model.model.load_state_dict(torch.load(os.path.join(path,'model_best_s2.pt')))

    args.model = 'RECOMPOSE'
    my_recomp = model.Model(args, checkpoint)
    my_recomp.model.load_state_dict(torch.load(os.path.join(path,'model_best_R.pt')))
    args.model = 'DISCRIMINATOR'
    my_dis = model.Model(args, checkpoint)
    my_dis.model.load_state_dict(torch.load(os.path.join(path,'model_best_D.pt')))

    args.n_colors = 3

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, my_model, my_recomp, my_dis, loss, checkpoint)
    while not t.terminate():
        t.test()

    checkpoint.done()


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
import torch

import utility
import data
import model
import loss
from option import args
from trainer_seg_hist_fuse import Trainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    my_model = model.Model(args, checkpoint)

    if os.path.basename(os.getcwd()) == 'SDSFusion':
        path = os.path.join(os.getcwd(),'pretrain/s1/model_best.pt')
    elif os.path.basename(os.getcwd()) == 'enhance_stage2':
        path1 = os.path.dirname(os.getcwd())
        path = os.path.join(path1,'pretrain/s1/model_best.pt')

    my_model.model.load_state_dict(torch.load(path), strict=False)

    args.model = 'RECOMPOSE'
    my_recomp = model.Model(args, checkpoint)

    args.model = 'DISCRIMINATOR'
    my_dis = model.Model(args, checkpoint)
    args.n_colors = 3

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, my_model, my_recomp, my_dis, loss, checkpoint)
    try:
        while not t.terminate():
            t.train()
            t.test()
    finally:
        t.close_writer()
        checkpoint.done()
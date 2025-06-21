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

    args.n_colors = 3

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, my_model, loss, checkpoint, adv=False)
    while not t.terminate():
        # t.train()
        t.test_paired()

    checkpoint.done()

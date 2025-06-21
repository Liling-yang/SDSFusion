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

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    my_model = model.Model(args, checkpoint)

    if os.path.basename(os.getcwd()) == 'SDSFusion':
        path = os.path.join(os.getcwd(),'pretrain/s2')
    elif os.path.basename(os.getcwd()) == 'enhance_stage1':
        path1 = os.path.dirname(os.getcwd())
        path = os.path.join(path1,'pretrain/s1')
    
    my_model.model.load_state_dict(torch.load(os.path.join(path,'model_best.pt')))

    args.n_colors = 3

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, my_model, loss, checkpoint, adv=True)

    while not t.terminate():
        t.test_unpaired()

    checkpoint.done()
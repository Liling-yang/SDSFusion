import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
import glob
from option import args
from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class LRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self._set_filesystem(args.dir_data)

        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_lr = self._scan()

        if args.ext.find('bin') >= 0:
            list_lr = self._scan()

            print('...check and load lr...')
            self.images_lr = [
                self._check_and_load(args.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]

        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_lr = list_lr

            elif args.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_lr.replace(self.apath, path_bin),
                    exist_ok=True
                )
              
                self.images_lr = []

                for l in list_lr:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr.append(b)

                    self._check_and_load(
                        args.ext, [l], b,  verbose=True, load=False
                    )
 
        if train:
            self.repeat = 3

    def _scan(self):
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )
        return names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if args.test_only == True:
            self.ext = ('.png', '.jpg')
        elif args.test_only == False:
            self.ext = ('.png','.png')

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR.pt'.format(self.split)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))

            b = []
            for _l in l:
                tmp_name = os.path.splitext(os.path.basename(_l))[0]
                tmp_image = _l

                tmp = {
                    'name': tmp_name,
                    'image': imageio.imread(tmp_image)
                }

                b.append(tmp)

            with open(f, 'wb') as _f: pickle.dump(b, _f) 
            return b

    def __getitem__(self, idx):
        lr, filename = self._load_file(idx)
        lr = self.get_patch(lr)
        lr = common.set_channel(lr, n_channels=self.args.n_colors)

        if isinstance(lr, list):
            lr = np.array(lr)
            lr = np.squeeze(lr, 0)

        lr_tensor = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
        if isinstance(lr_tensor, list):
            lr_tensor = torch.stack(lr_tensor)  # Ensure lr_tensor is a torch tensor
            lr_tensor = torch.squeeze(lr_tensor, 0)
        return lr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_lr) * self.repeat
        else:
            return len(self.images_lr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_lr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.images_lr[idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_lr['name']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_lr))
            if self.args.ext == 'img' or self.benchmark:
                lr = imageio.imread(f_lr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_lr, 'rb') as _f: lr = np.load(_f, allow_pickle=True)[0]['image']

        return lr, filename

    def get_patch(self, lr):
        if self.train:
            lr = common.get_patch(
                lr,
                patch_size=self.args.patch_size,
                scale=self.scale[self.idx_scale]
            )
            if not self.args.no_augment:
                lr = common.augment(lr)
        return lr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

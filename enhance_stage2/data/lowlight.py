import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
from data import srdata
import glob

class LowLight(srdata.SRData):
    def __init__(self, args, name='LowLight', train=True, benchmark=False):
        super(LowLight, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(LowLight, self)._set_filesystem(dir_data)

        path1 = os.getcwd()
        if os.path.basename(path1) == 'SDSFusion':
            self.apath = os.path.join(path1,'datasets','train','stage2')
        elif os.path.basename(path1) == 'enhance_stage2':
            path2 = os.path.dirname(path1)
            self.apath = os.path.join(path2,'datasets','train','stage2')
        print(self.apath)

        self.dir_hr   = os.path.join(self.apath, 'Normal') # jpg
        self.dir_lr   = os.path.join(self.apath, 'Low_degraded') # jpg
        self.dir_lrr = os.path.join(self.apath, 'Low_real_test_2_rs') # jpg
        self.dir_hq = os.path.join(self.apath, 'AVA_good_2') # jpg

    def _scan(self):
        names_hr, names_lr, names_lrr, names_hq = super(LowLight, self)._scan()
        names_hr   = names_hr[self.begin - 1:self.end]
        names_lr   = names_lr[self.begin - 1:self.end]
        names_lrr  = names_lrr[self.begin - 1:self.end]
        names_hq   = names_hq[self.begin - 1:self.end]

        return names_hr, names_lr, names_lrr, names_hq

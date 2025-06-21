import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定GPU运行
from data import srdata
import glob
from option import args

class LowLightTest(srdata.SRData):
    def __init__(self, args, name='LowLightTest', train=True, benchmark=False):
        super(LowLightTest, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(LowLightTest, self)._set_filesystem(dir_data)
        if os.path.basename(os.getcwd()) == 'SDSFusion':
            self.apath = os.path.join(os.getcwd(),'datasets/test')
        elif os.path.basename(os.getcwd()) == 'enhance_stage2':
            path1 = os.path.dirname(os.getcwd())
            self.apath = os.path.join(path1,'datasets/test')

        self.dir_hr = os.path.join(self.apath, 'Low_real_test_2_rs')
        self.dir_lr = os.path.join(self.apath, 'Low_real_test_2_rs')
        self.dir_hq = os.path.join(self.apath, 'AVA_good_2')
        self.dir_lrr = os.path.join(self.apath, 'Low_real_test_2_rs')


    def _scan(self):
        names_hr, names_lr, names_hq, names_lrr = super(LowLightTest, self)._scan()

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = names_lr[self.begin - 1:self.end]
        names_lrr = names_lrr[self.begin - 1:self.end]
        names_hq = names_hq[self.begin - 1:self.end]

        return names_hr, names_lr, names_lrr, names_hq

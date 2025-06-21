import os
from data import srdata
import glob

class LowLight(srdata.SRData):
    def __init__(self, args, name='LowLight', train=True, benchmark=False):
        super(LowLight, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(LowLight, self)._set_filesystem(dir_data)
        path1=os.getcwd()
        if os.path.basename(path1) == 'SDSFusion':
            self.apath = os.path.join(path1,'datasets','train','stage1')
        elif os.path.basename(path1) == 'enhance_stage1':
            path2 = os.path.dirname(path1)
            self.apath = os.path.join(path2,'datasets','train','stage1')
        print(self.apath)
        self.dir_hr   = os.path.join(self.apath, 'Our_normal')
        self.dir_lr   = os.path.join(self.apath, 'Our_low')

    def _scan(self):
        names_hr, names_lr = super(LowLight, self)._scan()
        names_hr   = names_hr[self.begin - 1:self.end]
        names_lr   = names_lr[self.begin - 1:self.end]

        return names_hr, names_lr

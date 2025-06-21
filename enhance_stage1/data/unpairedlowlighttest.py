import os
from data import lrdata
import glob

class unpairedlowlighttest(lrdata.LRData):
    def __init__(self, args, name='LowLightTest', train=True, benchmark=False):
        super(unpairedlowlighttest, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):

        super(unpairedlowlighttest, self)._set_filesystem(dir_data)
        path1 = os.getcwd()
        if os.path.basename(path1) == 'SDSFusion':
            self.apath = os.path.join(path1,'datasets','test','LLVIP','vi')
        elif os.path.basename(path1) == 'enhance_stage1':
            path2 = os.path.dirname(path1)
            self.apath = os.path.join(path2,'datasets','test','LLVIP','vi')
        print(self.apath)
        self.dir_lr = self.apath
        
        


    def _scan(self):
        names_lr = super(unpairedlowlighttest, self)._scan()
        names_lr = names_lr[self.begin - 1:self.end]
        return names_lr

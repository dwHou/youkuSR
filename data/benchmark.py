import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import glob
import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in glob.glob(self.dir_hr + '/*' + self.ext):
            filename = os.path.splitext(os.path.basename(entry))[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join( self.dir_lr, filename + self.ext))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'x{}'.format(self.scale[self.idx_scale]), 'hr')
        self.dir_lr = os.path.join(self.apath, 'x{}'.format(self.scale[self.idx_scale]), 'lr')
        self.ext = '.png'

import os

from data import srdata
from data import common

import random
import numpy as np
import scipy.misc as misc


class Youku_e(srdata.SRData):
    def __init__(self, args, train=True):
        super(Youku_e, self).__init__(args, train)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.offset_val
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin, idx_end):
            hr_dirname = os.path.join(self.dir_hr, 'Youku_{:0>5}_h_GT'.format(i))
            lr_dirname = os.path.join(self.dir_lr, 'Youku_{:0>5}'.format(i))
            for i in range(1, self.n_f + 1):
                hr = hr_dirname + '/' + '{:0>3}'.format(i) + self.ext
                lr = lr_dirname + '/' + '{:0>3}'.format(i) + self.ext
                list_hr.append(hr)
                list_lr[self.idx_scale].append(lr)

        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            if not self.args.not_hr:
                hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            if not self.args.not_hr:
                hr = np.load(hr)
        else:
            filename = str(idx + 1)

        target_index = int(os.path.split(filename)[-1][:3])
        # filename = os.path.splitext(os.path.split(filename)[-1])[0]
        filename = os.path.split(os.path.split(filename)[0])[-1]
        filename = filename[0:12] + '{:0>3}'.format(target_index)
        if self.args.not_hr:
            # for compatibility
            hr = np.ndarray(lr.shape)

        return lr, hr, filename

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/youku/png'
        self.ext = '.png'
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'hr')
            self.dir_lr = '../test/EDVR-median-dataset/results/'
            # self.dir_hr = os.path.join(self.apath, 'train_hr')
            # self.dir_lr = os.path.join(self.apath, 'train_lr')
            self.n_f = 100 # read n frames from a sample
        else:
            self.dir_hr = os.path.join(self.apath, 'hr')
            # self.dir_lr = os.path.join(self.apath, 'lr')
            self.dir_lr = '../test/EDVR-median-dataset/results/'
            self.n_f = self.args.n_test_frames # read n frames from a sample


    def __len__(self):
        if self.train:
            return len(self.images_hr)
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _get_patch(self, lr, hr):
        ps = self.args.patch_size
        multi_frames = len(lr.shape) == 4
        if self.train:
            h, w = lr.shape[-3:-1]
            x = random.randrange(0, w - ps + 1)
            y = random.randrange(0, h - ps + 1)
            if multi_frames:
                lr, hr = lr[:, y:y + ps, x:x + ps, :], hr[:, y:y + ps, x:x + ps, :]
            else:
                lr, hr = lr[y:y + ps, x:x + ps, :], hr[y:y + ps, x:x + ps, :]
            lr, hr = common.augment([lr, hr])
        return lr, hr

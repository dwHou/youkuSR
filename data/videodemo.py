import os

from data import common

import numpy as np
import scipy.misc as misc
import torch.utils.data as data
import glob


class VideoDemo(data.Dataset):
    def __init__(self, args, train=False):
        super(VideoDemo, self).__init__()
        self.args = args
        self.train = train
        self.idx_scale = 0
        self.video_path = sorted(glob.glob(args.dir_demo + '/*'))
        self.img_path = []
        self.video_len = {}
        for path in self.video_path:
            imgs = sorted(glob.glob(path + '/*'))
            self.img_path.extend(imgs)
            self.video_len[path] = len(imgs)

    def __getitem__(self, idx):
        lr_path = self.img_path[idx]
        video_path = os.path.dirname(lr_path)
        video_name = os.path.basename(video_path)
        frame_idx = int(os.path.basename(lr_path).split('.')[0])
        filename = video_name + '_{}'.format(frame_idx)
        idxs = self.index_generation(frame_idx, self.video_len[video_path], self.args.n_frames)
        lrs = []
        for idx in idxs:
            img_path = os.path.join(os.path.dirname(lr_path), '{}.png'.format(idx))
            temp = misc.imread(img_path)
            lrs.append(temp)
        lrs = np.array(lrs)

        lrs = common.set_channel([lrs], self.args.n_colors)[0]
        lrs = common.np2Tensor([lrs], self.args.rgb_range)[0]

        return lrs, -1, filename


    def __len__(self):
        return len(self.img_path)

    def index_generation(self, crt_i, max_n, N):
        max_n = max_n
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 1:
                add_idx = 1
            elif i > max_n:
                add_idx = max_n
            else:
                add_idx = i
            return_l.append(add_idx)
        return return_l


    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

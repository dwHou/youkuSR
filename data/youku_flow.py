import os
import random

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
import pyflow

class youku(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.scale = args.scale
        self.idx_scale = 0

        # self.repeat = args.test_every // (args.n_train // args.batch_size)

        self._set_filesystem(args.dir_data)

        if args.ext == 'img':
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            raise NotImplementedError
        elif args.ext.find('bin') >= 0:
            raise NotImplementedError
        else:
            print('Please define data type')


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
            # hr_dirname = 'Youku_{:0>5}_h_GT'.format(i)
            # lr_dirname = 'Youku_{:0>5}_l'.format(i)
            # hr = []
            # lr = []
            # for i in range(1, 101):
                # relatepath_hr = hr_dirname + '/' + '{:0>3}'.format(i) + self.ext
                # relatepath_lr = lr_dirname + '/' + '{:0>3}'.format(i) + self.ext
                # hr.append(os.path.join(self.dir_hr, relatepath_hr))
                # lr.append(os.path.join(self.dir_hr, relatepath_lr))

            # list_hr.append(hr)
            # list_lr[0].append(lr)

            hr_dirname = os.path.join(self.dir_hr, 'Youku_{:0>5}_h_GT'.format(i))
            lr_dirname = os.path.join(self.dir_lr, 'Youku_{:0>5}_l'.format(i))
            for i in range(1, self.n_f + 1):
                hr = hr_dirname + '/' + '{:0>3}'.format(i) + self.ext
                lr = lr_dirname + '/' + '{:0>3}'.format(i) + self.ext
                list_hr.append(hr)
                list_lr[self.idx_scale].append(lr)

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/youku/png'
        self.ext = '.png'
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'train_hr')
            self.dir_lr = os.path.join(self.apath, 'train_lr')
            self.n_f = 100 # read n frames from a sample
        else:
            self.dir_hr = os.path.join(self.apath, 'valid_hr')
            self.dir_lr = os.path.join(self.apath, 'valid_lr')
            self.n_f = self.args.n_test_frames # read n frames from a sample


    def __len__(self):
        if self.train:
            return len(self.images_hr)
            # return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def __getitem__(self, idx):
        lr, hr, neigbors, filename = self._load_file(idx)
        lr, hr, neigbors = self._get_patch(lr, hr, neigbors)
        lr, hr, neigbors = self._augment(lr, hr, neigbors)
        lr, hr, neigbors = self._makecontiguous(lr, hr, neigbors)
        flows = [self._get_flow(lr, j) for j in neigbors]
        flows_tensor = [torch.from_numpy(flow.transpose(2, 0, 1)).float() for flow in flows]
        lr_tensor, hr_tensor, neigbors_tensor = self._np2Tensor(lr, hr, neigbors, self.args.rgb_range)

        #flows neigbors list to tensor
        flows_tensor = torch.cat([j.unsqueeze(dim=0) for j in flows_tensor])
        neigbors_tensor = torch.cat([j.unsqueeze(dim=0) for j in neigbors_tensor])

        #save img for test
        # def _saveimg(filename, img):
            # ndarr = img.byte().permute(1, 2, 0).cpu().numpy()
            # misc.imsave('{}.png'.format(filename), ndarr)
        # _saveimg('testoutput/'+ filename+'lr', lr_tensor)
        # _saveimg('testoutput/'+ filename+'hr', hr_tensor)
        # for i in range(6):
            # _saveimg('testoutput/'+ filename+'neigbors'+str(i), neigbors_tensor[i])
        # import pdb; pdb.set_trace()  # XXX BREAKPOINT

        return lr_tensor, hr_tensor, neigbors_tensor, flows_tensor, filename

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        n_frames = self.args.n_frames

        if self.args.ext == 'img':
            filename = hr
            lr_dir = os.path.split(lr)[0]
            lr = misc.imread(lr)
            hr = misc.imread(hr)

            target_index = int(os.path.split(filename)[-1][:3])
            if target_index <= n_frames // 2:
                neigbors_index = [x for x in range(1, n_frames + 1) if x != target_index]
            elif self.n_f - target_index <= n_frames // 2:
                neigbors_index = [x for x in range(self.n_f+1-n_frames, self.n_f+1) if x != target_index]
            else:
                neigbors_index = [target_index-i for i in range(1, n_frames//2 + 1)]
                neigbors_index.extend([target_index+i for i in range(1, n_frames//2 + 1)])
            neigbors = []
            for i in neigbors_index:
                neigbor = misc.imread(os.path.join(lr_dir, '{:0>3}'.format(i) + self.ext))
                neigbors.append(neigbor)
            neigbors = np.array(neigbors)

        elif self.args.ext.find('sep') >= 0:
            raise NotImplementedError
        else:
            filename = str(idx + 1)

        filename = os.path.split(os.path.split(filename)[0])[-1]
        filename = filename[0:12] + '{:0>3}'.format(target_index)

        return lr, hr, neigbors, filename

    def _get_patch(self, lr, hr, neigbors):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            ih, iw = lr.shape[0:2]
            tp = patch_size
            ip = tp // scale

            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)
            tx, ty = scale * ix, scale * iy

            lr = lr[iy:iy + ip, ix:ix + ip, :]
            hr = hr[ty:ty + tp, tx:tx + tp, :]
            neigbors = neigbors[:, iy:iy + ip, ix:ix + ip, :]

        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr, neigbors

    def _augment(self, lr, hr, neigbors, hflip=True, rot=True):
        if not self.train:
            return lr, hr, neigbors
        vflip = rot and random.random() < 0.5
        hflip = hflip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _f(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

            return img

        return _f(lr), _f(hr), [_f(j) for j in neigbors]

        # lr = _f(lr)
        # hr = _f(hr)
        # for i in range(len(neigbors)):
            # neigbors[i] = _f(neigbors[i])
        # return lr, hr, neigbors

    def _makecontiguous(self, lr, hr, neigbors):
        def _f(img):
            i = np.ascontiguousarray(img)
            return i

        return _f(lr), _f(hr), [_f(j) for j in neigbors]

    def _np2Tensor(self, lr, hr, neigbors, rgb_range):
        def _f(img):
            # np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            np_transpose = img.transpose(2, 0, 1)
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)

            return tensor

        return _f(lr), _f(hr), [_f(j) for j in neigbors]

    def _get_flow(self, im1, im2):
        im1 = np.array(im1)
        im2 = np.array(im2)
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        #flow = rescale_flow(flow,0,1)
        return flow.astype('float32')


    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

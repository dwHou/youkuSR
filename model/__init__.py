import os
from importlib import import_module

import torch
import torch.nn as nn
import functools


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.test_patch_size = args.test_patch_size
        self.chop_threshold = args.chop_threshold
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        if args.print_model: print(self.model)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        if self.test_patch_size != 0:
            forward1 = functools.partial(self.forward_patch, forward_function=self.model)
        else:
            forward1 = self.model
        if self.chop:
            forward2 = functools.partial(self.forward_chop, forward_function=forward1)
        else:
            forward2 = forward1
        if self.self_ensemble:
            forward3 = functools.partial(self.forward_x8, forward_function=forward2)
        else:
            forward3 = forward2
        return forward3(x)

        # if self.self_ensemble and not self.training:
            # if self.test_patch_size != 0:
                # forward_function = self.forward_patch
            # elif self.chop:
                # forward_function = self.forward_chop
            # else:
                # forward_function = self.model.forward

            # return self.forward_x8(x, forward_function)
        # elif self.test_patch_size != 0 and not self.training:
            # return self.forward_patch(x)
        # elif self.chop and not self.training:
            # return self.forward_chop(x)
        # else:
            # return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_patch(self,x, forward_function, shave=4):
        multi_frame = len(x.size()) == 5
        if multi_frame:
            b, f, c, h, w = x.size()
        else:
            b, c, h, w = x.size()
        scale = self.scale[self.idx_scale]
        test_patch = self.test_patch_size // scale
        interval = test_patch - 2 * shave
        nh = (h - 2 * shave) // interval
        nw = (w - 2 * shave) // interval
        rh = ((h - 2 * shave) % interval) + shave
        rw = ((w - 2 * shave) % interval) + shave
        sps = []  # interval patch top_left points
        for i in range(nh):
            for j in range(nw):
                sps.append((shave + interval * i, shave + interval * j))
            if rw != shave:
                sps.append((shave + interval * i, w - (shave + interval)))
        if rh != shave:
            for j in range(nw):
                sps.append((h - (shave + interval), shave + interval * j))
            if rw != shave:
                sps.append((h - (shave + interval), w - (shave + interval)))


        lr_list = []
        if multi_frame:
            for sp in sps:
                px, py = sp[0] - shave, sp[1] - shave
                lr_list.append(x[:, :, :, px:px + test_patch, py:py + test_patch])
        else:
            for sp in sps:
                px, py = sp[0] - shave, sp[1] - shave
                lr_list.append(x[:, :, px:px + test_patch, py:py + test_patch])
        lr = torch.cat(lr_list, dim=0)

        sr = forward_function(lr)
        sr_list = []
        for i in range(len(lr_list)):
            sr_list.append(sr[i * b : (i + 1) * b])

        output = x.new(b, c, h * scale, w * scale)
        lr_nh = nh + 1 if rh != shave else nh
        lr_nw = nw + 1 if rw != shave else nw
        sx = shave * scale
        invx = interval * scale
        rwx = rw * scale
        rhx = rh * scale

        # corner top-left top-right bottom-left bottom-right
        output[:, :, :sx + invx, :sx + invx] = \
            sr_list[0][:, :, :sx + invx, :sx + invx]
        output[:, :, :sx + invx, -rwx:] = \
            sr_list[lr_nw - 1][:, :, :sx + invx, -rwx:]
        output[:, :, -rhx:, :sx + invx] = \
            sr_list[lr_nw * (lr_nh - 1)][:, :, -rhx:, :sx + invx]
        output[:, :, -rhx:, -rwx:] = \
            sr_list[-1][:, :, -rhx:, -rwx:]

        # left right
        for i in range(1, lr_nh - 1):
            output[:, :, sx + i * invx:sx + (i+1) * invx, :sx + invx] = \
                sr_list[i * lr_nw][:, :, sx:-sx, :sx + invx]
            output[:, :, sx + i * invx:sx + (i+1) * invx, -rwx:] = \
                sr_list[(i+1) * lr_nw - 1][:, :, sx:-sx, -rwx:]
        # top bottom
        for i in range(1, lr_nw - 1):
            output[:, :, :sx + invx, sx + i * invx:sx + (i+1) *invx] = \
                sr_list[i][:, :, :sx + invx, sx:-sx]
            output[:, :, -rhx:, sx + i * invx:sx + (i+1) * invx] = \
                sr_list[(lr_nh - 1) * lr_nw + i][:, :, -rhx:, sx:-sx]

        # center
        for i in range(1, lr_nh - 1):
            for j in range(1, lr_nw - 1):
                output[:, :, sx + i * invx:sx + (i+1) * invx, sx + j * invx:sx + (j+1) * invx] = \
                    sr_list[i * lr_nw + j][:, :, sx:-sx, sx:-sx]

        return output


    def forward_chop(self, x, forward_function, shave=10, min_size=0):
        min_size = self.chop_threshold
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        multi_frame = len(x.size()) == 5
        if multi_frame:
            b, f, c, h, w = x.size()
        else:
            b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        if multi_frame:
            lr_list = [
                x[:, :, :, 0:h_size, 0:w_size],
                x[:, :, :, 0:h_size, (w - w_size):w],
                x[:, :, :, (h - h_size):h, 0:w_size],
                x[:, :, :, (h - h_size):h, (w - w_size):w]]
        else:
            lr_list = [
                x[:, :, 0:h_size, 0:w_size],
                x[:, :, 0:h_size, (w - w_size):w],
                x[:, :, (h - h_size):h, 0:w_size],
                x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = forward_function(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, forward_function, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        def _transform_mulitframe(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 2, 4, 3)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        batch_size = x.size()[0]
        lr_list = [x]
        for tf in 'v', 'h', 't':
            if len(x.shape) == 5:
                lr_list.extend([_transform_mulitframe(t, tf) for t in lr_list])
            else:
                lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        # lr_list_1 = torch.cat(lr_list[:4], dim=0)
        # lr_list_transpose = torch.cat(lr_list[4:], dim=0)
        # sr_list_1 = forward_function(lr_list_1)
        # sr_list_transpose = forward_function(lr_list_transpose)
        # sr_list = [*sr_list_1.unsqueeze(1), *sr_list_transpose.unsqueeze(1)]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output = []
        for i in range(batch_size):
            single_img = []
            for j in range(8):
                single_img.append(sr_list[j][i].unsqueeze(0))
            output.append(torch.cat(single_img, dim=0).mean(dim=0, keepdim=True))
        output = torch.cat(output, dim=0)

        # output_cat = torch.cat(sr_list, dim=0)
        # output = output_cat.mean(dim=0, keepdim=True)

        return output


import torch


import utility
import data
import model
from tqdm import tqdm
from option import args



class Evaluator():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = my_model
        self.global_timer = utility.timer()

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                sum_psnr = 0
                img_count = 0
                for idx_batch, (lr, hr, filename, _) in enumerate(tqdm_test):
                    # batch_timer = utility.timer()
                    if self.args.n_test_frames == 0 or img_count % self.args.n_test_frames == 0:
                        timer_test.tic()
                    # no_eval = (hr.nelement() == 1)
                    no_eval = self.args.not_hr
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    # self.ckp.write_log(
                        # 'Forward time: {:.2f} s\n'.format(batch_timer.toc()), refresh=True
                    # )

                    for idx_sr in range(len(sr)):
                        if self.args.n_test_frames == 0 or img_count % self.args.n_test_frames == 0:
                            sum_psnr = 0
                        img_count += 1
                        img_filename = filename[idx_sr]
                        if not no_eval:
                            psnr = utility.calc_psnr(
                                sr[idx_sr].unsqueeze(dim=0), hr[idx_sr], scale, self.args.rgb_range,
                                psnr_matlab=self.args.psnr_matlab
                            )
                            sum_psnr += psnr
                            self.ckp.write_log(
                                '[{} x{}]\tPSNR: {:.3f}'.format(
                                    img_filename,
                                    scale,
                                    psnr, refresh=False
                                )
                            )
                            if self.args.n_test_frames != 0 and img_count % self.args.n_test_frames == 0:
                                self.ckp.write_log(
                                    '[{} x{}]\tPSNR: {:.3f}'.format(
                                        img_filename[:-4],
                                        scale,
                                        sum_psnr / self.args.n_test_frames, refresh=False
                                    )
                                )

                        if self.args.n_test_frames != 0 and img_count % self.args.n_test_frames == 0:
                            self.ckp.write_log(
                                'Test time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
                            )

                        if self.args.save_results:
                            # save_test_SR need a 4-dim vector [B,C,H,W] whose batch size == 1
                            sr_save = sr[idx_sr].unsqueeze(dim=0)
                            self.ckp.save_test_SR(img_filename, sr_save, None, scale)


        self.ckp.write_log(
            'Total time: {:.2f}min\n'.format(self.global_timer.toc() / 60), refresh=True
        )

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        t = Evaluator(args, loader, model, checkpoint)
        t.test()

        checkpoint.done()

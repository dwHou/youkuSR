from decimal import Decimal

import utility

import torch
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer, self).__init__(args, loader, my_model, my_loss, ckp)

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        self.ckp.write_log('whole time:{} min'.format(self.global_timer.toc() / 60))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        best = self.ckp.log.max(0)
        # evaluate model every valid_interval epoches
        if epoch % self.args.valid_interval == 0:
            self.ckp.write_log('\nEvaluation:')
            self.model.eval()

            timer_test = utility.timer()
            with torch.no_grad():
                for idx_scale, scale in enumerate(self.scale):
                    eval_acc = 0
                    self.loader_test.dataset.set_scale(idx_scale)
                    tqdm_test = tqdm(self.loader_test, ncols=80)
                    for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                        no_eval = (hr.nelement() == 1)
                        if not no_eval:
                            lr, hr = self.prepare([lr, hr])
                        else:
                            lr = self.prepare([lr])[0]

                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        if not no_eval:
                            eval_acc += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range
                            )

                        if self.args.save_results:
                            for idx_sr in range(len(sr)):
                                filename = filename[idx_sr]
                                # save_test_SR need a 4-dim vector [B,C,H,W] whose batch size == 1
                                sr_save = sr[idx_sr].unsqueeze(dim=0)
                                self.ckp.save_test_SR(filename, sr_save, epoch, scale)

                    self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
            self.ckp.write_log(
                'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )

        # save models
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
            self.ckp.plot_psnr(epoch)


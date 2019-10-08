from decimal import Decimal

import utility

import torch
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
        self.ckp.save(self, epoch)

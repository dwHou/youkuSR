import torch

import utility
import data
import model
import loss
import trainer
from option import args

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = trainer.Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()


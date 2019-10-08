from importlib import import_module


class Trainer:
    def __init__(self, args, loader, model, loss, checkpoint):
        trainer = import_module('trainer.' + args.trainer.lower())
        self.trainer = trainer.Trainer(args, loader, model, loss, checkpoint)

    def train(self):
        self.trainer.train()

    def test(self):
        self.trainer.test()

    def terminate(self):
        return self.trainer.terminate()

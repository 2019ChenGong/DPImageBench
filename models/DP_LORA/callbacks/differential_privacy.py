from pytorch_lightning.callbacks import Callback


class DPCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        epsilon = pl_module.get_epsilon_spent()
        print("#################################################################",epsilon)
        # self.log('epsilon', epsilon, prog_bar=True, logger=True, on_epoch=False)

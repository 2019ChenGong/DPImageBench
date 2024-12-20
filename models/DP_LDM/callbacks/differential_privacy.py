from pytorch_lightning.callbacks import Callback


class DPCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        epsilon = pl_module.get_epsilon_spent(batch_idx)
        self.log('epsilon', epsilon, prog_bar=True, logger=True, on_step=True, on_epoch=False)

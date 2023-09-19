"""Basic LightningModules on which other modules can be built."""
import lightning.pytorch as pl

import torch
from torch import nn
from torchmetrics import Accuracy
from typing import Callable

class LitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, loss_fn: Callable):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.acc_fn = Accuracy(task="multiclass", num_classes=10)
        self.save_hyperparameters(ignore=['model'])

    def configure_optimizers(self):
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "epoch", "frequency": 1, "monitor": "val/loss"}}

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)
    
    def _run_on_batch(self, batch, with_preds=False):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return x, y, logits, loss

    def training_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        accu = self.acc_fn(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/accu", accu, on_step=True, on_epoch=False, prog_bar=True)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        accu = self.acc_fn(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val/accu", accu, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        accu = self.acc_fn(logits, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/accu", accu, on_step=False, on_epoch=True)


def main():
    pass

if __name__ == '__main__':
    main()


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Liu2Net(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            torch.nn.Linear(784, 512),
            nn.ReLU(),

            torch.nn.Linear(512, 256),
            nn.ReLU(),

            torch.nn.Linear(256, 128),
            nn.ReLU(),

            torch.nn.Linear(128, 64),
            nn.ReLU(),

            torch.nn.Linear(64, num_classes),

        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(-1, 784)
        return self.net(x)


class Liu2MNIST(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=0.0,
        num_classes=10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator =Liu2Net(num_classes=num_classes)

    def forward(self, x):
        logits = self.generator(x)
        return logits

    def _shared_step(self, batch, stage):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, dim=1)

        return {
            "pred": pred,
            "target": y,
            "logits": logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer
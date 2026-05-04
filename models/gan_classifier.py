import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class PerturbationGenerator(nn.Module):
    """
    输入 MNIST 图像，生成一个小扰动。
    """

    def __init__(self, perturb_eps=0.3):
        super().__init__()
        self.perturb_eps = perturb_eps

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        delta = self.net(x)
        delta = self.perturb_eps * delta
        return delta


class ClassifierDiscriminator(nn.Module):
    """
    同时做两件事：
    1. 分类：输出 10 类 logits
    2. 判别：判断输入是真实样本还是 Generator 扰动样本
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.LeakyReLU(0.2),
        )

        self.cls_head = nn.Linear(128, num_classes)
        self.adv_head = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.feature(x)
        cls_logits = self.cls_head(feat)
        adv_logits = self.adv_head(feat).squeeze(1)
        return cls_logits, adv_logits


class GANClassifierMNIST(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=0.0,
        num_classes=10,
        adv_weight=0.2,
        perturb_eps=0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = PerturbationGenerator(
            perturb_eps=perturb_eps,
        )
        self.discriminator = ClassifierDiscriminator(
            num_classes=num_classes,
        )

        self.automatic_optimization = False

    def forward(self, x):
        cls_logits, _ = self.discriminator(x)
        return cls_logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        opt_g, opt_d = self.optimizers()

        # =========================
        # 1. Train Discriminator
        # =========================
        delta = self.generator(x).detach()
        x_fake = x + delta

        real_cls_logits, real_adv_logits = self.discriminator(x)
        fake_cls_logits, fake_adv_logits = self.discriminator(x_fake)

        cls_loss_real = F.cross_entropy(real_cls_logits, y)
        cls_loss_fake = F.cross_entropy(fake_cls_logits, y)

        real_label = torch.ones_like(real_adv_logits)
        fake_label = torch.zeros_like(fake_adv_logits)

        adv_loss_real = F.binary_cross_entropy_with_logits(
            real_adv_logits,
            real_label,
        )
        adv_loss_fake = F.binary_cross_entropy_with_logits(
            fake_adv_logits,
            fake_label,
        )

        d_loss = (
            cls_loss_real
            + cls_loss_fake
            + self.hparams.adv_weight * (adv_loss_real + adv_loss_fake)
        )

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # =========================
        # 2. Train Generator
        # =========================
        delta = self.generator(x)
        x_fake = x + delta

        fake_cls_logits, fake_adv_logits = self.discriminator(x_fake)

        cls_loss_g = F.cross_entropy(fake_cls_logits, y)

        fool_label = torch.ones_like(fake_adv_logits)
        adv_loss_g = F.binary_cross_entropy_with_logits(
            fake_adv_logits,
            fool_label,
        )

        g_loss = cls_loss_g + self.hparams.adv_weight * adv_loss_g

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        with torch.no_grad():
            pred = torch.argmax(real_cls_logits, dim=1)
            acc = (pred == y).float().mean()

        self.log("train_d_loss", d_loss, prog_bar=True, sync_dist=True)
        self.log("train_g_loss", g_loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

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
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return [opt_g, opt_d]
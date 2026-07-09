from pathlib import Path

import pytorch_lightning as pl
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from data.mnist_datamodule import MNISTDataModule
from models import get_model
from utils.callbacks import get_callbacks
from utils.misc import get_exp_name


def get_datamodule(conf):
    return MNISTDataModule(**conf.data)


def train(conf):
    pl.seed_everything(conf.system.seed, workers=True)

    exp_name = get_exp_name(conf)

    tb_logger = TensorBoardLogger(
        save_dir=conf.system.log_dir,
        name=exp_name,
        version=conf.version,
        default_hp_metric=False,
    )

    root_dir = str(Path(conf.root_dir) / exp_name)
    logger.info("root_dir: " + root_dir)

    callbacks = get_callbacks(conf)
    model = get_model(conf)
    datamodule = get_datamodule(conf)

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        logger=tb_logger,
        callbacks=callbacks,
        **conf.trainer,
        # accelerator='gpu',devices=1,max_epochs=20,strategy='auto',log_every_n_steps=10,deterministic=True
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=conf.ckpt)


def validate(conf):
    model = get_model(conf)
    datamodule = get_datamodule(conf)

    trainer = pl.Trainer(**conf.trainer)
    trainer.validate(model, datamodule=datamodule, ckpt_path=conf.ckpt)


def test(conf):
    model = get_model(conf)
    datamodule = get_datamodule(conf)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        **conf.trainer
    )
    trainer.test(model, datamodule=datamodule, ckpt_path=conf.ckpt)


def predict(conf):
    model = get_model(conf)
    datamodule = get_datamodule(conf)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        **conf.trainer
    )
    outputs = trainer.predict(model, datamodule=datamodule, ckpt_path=conf.ckpt)

    if trainer.global_rank == 0:
        logger.info(f"num prediction batches: {len(outputs)}")
    logger.info(outputs.sha)


if __name__ == "__main__":
    conf = OmegaConf.create(
        {
            "conf": "conf/config.yaml",
        }
    )
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    # logger.info(OmegaConf.to_yaml(conf))
    eval(conf.cmd)(conf)
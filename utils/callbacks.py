from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def get_callbacks(conf):
    monitor = conf.callbacks.monitor
    mode = conf.callbacks.mode

    callbacks = []

    ckpt_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=conf.callbacks.save_top_k,
        save_last=True,
        filename="{epoch:03d}-{" + monitor + ":.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=conf.callbacks.patience,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks.append(ckpt_callback)
    callbacks.append(early_stop_callback)
    callbacks.append(lr_monitor)

    return callbacks
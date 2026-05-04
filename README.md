# MNIST Lightning (PyTorch Lightning 2.2.5)

基于 PyTorch Lightning 的 MNIST 手写数字识别示例工程，包含两套模型：

- GeneratorOnlyMNIST：普通分类模型（单生成器）
- GANClassifierMNIST：带对抗扰动的分类模型（Generator + Discriminator）

支持：
- trainer.fit / validate / test / predict
- OmegaConf 参数管理（CLI + YAML）
- TensorBoard 日志
- ModelCheckpoint + EarlyStopping
- GPU / DDP

---

## 1. 环境配置

```bash
conda create -n mnist python=3.10.16
conda activate mnist
pip install -r requirements.txt
````

---

## 2. 训练

### 2.1 GeneratorOnlyMNIST

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=train \
    conf=conf/config.yaml \
    model.name=GeneratorOnlyMNIST \
    version=g0
```

### 2.2 GANClassifierMNIST

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=train \
    conf=conf/config.yaml \
    model.name=GANClassifierMNIST \
    version=g1
```

日志与模型输出路径：

```text
logs/{ModelName}/{version}/
```

示例：

```text
logs/GeneratorOnlyMNIST/g0/
logs/GANClassifierMNIST/g1/
```

---

## 3. 验证 / 测试 / 预测

### 3.1 GeneratorOnlyMNIST

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=validate \
    conf=conf/config.yaml \
    ckpt=logs/GeneratorOnlyMNIST/g0/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=test \
    conf=conf/config.yaml \
    ckpt=logs/GeneratorOnlyMNIST/g0/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=predict \
    conf=conf/config.yaml \
    ckpt=logs/GeneratorOnlyMNIST/g0/checkpoints/last.ckpt
```

### 3.2 GANClassifierMNIST

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=validate \
    conf=conf/config.yaml \
    ckpt=logs/GANClassifierMNIST/g1/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=test \
    conf=conf/config.yaml \
    ckpt=logs/GANClassifierMNIST/g1/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0 python -m main \
    cmd=predict \
    conf=conf/config.yaml \
    ckpt=logs/GANClassifierMNIST/g1/checkpoints/last.ckpt
```

---

## 4. TensorBoard

```bash
tensorboard --host 0.0.0.0 --port 8888 --logdir logs/
```

访问：

```text
http://<server_ip>:8888
```

端口需大于 1024。

---

## 5. 配置说明

主要配置文件：

```text
conf/config.yaml
```

关键字段：

* `cmd`：train / validate / test / predict
* `model.name`：模型选择
* `trainer`：Lightning Trainer 参数（GPU / DDP / epoch 等）
* `callbacks`：EarlyStopping / ModelCheckpoint

CLI 参数会覆盖 YAML，例如：

```bash
model.lr=0.0005 trainer.max_epochs=30
```

---

## 6. 多卡训练（DDP）

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m main \
    cmd=train \
    conf=conf/config.yaml \
    model.name=GeneratorOnlyMNIST \
    trainer.devices=2 \
    trainer.strategy=ddp
```

---

## 7. 输出结构

```text
logs/
└── {ModelName}/
    └── {version}/
        ├── checkpoints/
        │   ├── epoch=xxx-val_acc=xxxx.ckpt
        │   └── last.ckpt
        ├── events.out.tfevents...
        └── hparams.yaml
```

---

## 8. 注意事项

* `test / validate / predict` 默认会生成 `lightning_logs/`，如需关闭需在 Trainer 中设置 `logger=False`
* 使用 GPU 时需保证 CUDA 与 PyTorch 版本匹配
* Windows / WSL 环境建议关闭 `num_workers` 或设为较小值


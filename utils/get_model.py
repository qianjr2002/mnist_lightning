import models

def get_model(conf):
    name = conf.model.name
    kwargs = {
        "lr": conf.model.lr,
        "weight_decay": conf.model.weight_decay,
        "num_classes": conf.model.num_classes,
    }
    model = getattr(models, name)(**kwargs)
    return model
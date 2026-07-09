from models.generator_only import GeneratorOnlyMNIST
from models.gan_classifier import GANClassifierMNIST
from models.liu2_mnist import Liu2MNIST

def get_model(conf):
    name = conf.model.name

    if name == "GeneratorOnlyMNIST":
        return GeneratorOnlyMNIST(
            lr=conf.model.lr,
            weight_decay=conf.model.weight_decay,
            num_classes=conf.model.num_classes,
        )

    if name == "GANClassifierMNIST":
        return GANClassifierMNIST(
            lr=conf.model.lr,
            weight_decay=conf.model.weight_decay,
            num_classes=conf.model.num_classes,
            adv_weight=conf.model.gan.adv_weight,
            perturb_eps=conf.model.gan.perturb_eps,
        )
    
    if name == "Liu2MNIST":
        return Liu2MNIST(
            lr=conf.model.lr,
            weight_decay=conf.model.weight_decay,
            num_classes=conf.model.num_classes,
    )

    raise ValueError(f"Unknown model name: {name}")
from .cifar10 import CIFARDataLoader
from .tiny_imagenet import TinyImageNetDataLoader
from .celeba import CelebADataLoader
from .af_hq import AFHQDataLoader
from .flowers_102 import Flowers102DataLoader

DATASET_REGISTRY = {
    "cifar10": CIFARDataLoader,
    "tiny_imagenet": TinyImageNetDataLoader,
    "celeba": CelebADataLoader,
    "afhq": AFHQDataLoader,
    "flowers102": Flowers102DataLoader,
}


def get_dataset_loader(name: str):
    """Return the dataset loader class for the given config name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]

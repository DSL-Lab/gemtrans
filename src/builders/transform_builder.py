from torchvision.transforms import (
    ToTensor,
    Compose,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    Grayscale,
)
from torchvision.transforms._transforms_video import (
    RandomResizedCropVideo,
    RandomHorizontalFlipVideo,
)


def build(config, train):
    aug_transform = None

    if config.mode == "pretrain":
        transform = Compose(
            [Grayscale(), Resize((config.frame_size, config.frame_size))]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Resize((config.frame_size, config.frame_size)),
                Normalize((config.mean), (config.std)),
            ]
        )

    if config.mode == "as":
        aug_transform = Compose(
            [
                RandomResizedCropVideo(size=config.frame_size, scale=(0.7, 1)),
                RandomHorizontalFlipVideo(p=0.3),
            ]
        )
    elif config.mode == "ef":
        aug_transform = Compose([RandomHorizontalFlip(p=0.3)])

    return transform, aug_transform

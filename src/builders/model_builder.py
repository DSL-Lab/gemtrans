from src.core.model import SpaceTimeFactorizedViViT


def build(config):
    return SpaceTimeFactorizedViViT(**config)

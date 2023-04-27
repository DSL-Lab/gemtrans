from torch.optim import lr_scheduler


def build(optimizer, config):

    return lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=config.patience,
        threshold=config.threshold,
        min_lr=config.min_lr,
    )

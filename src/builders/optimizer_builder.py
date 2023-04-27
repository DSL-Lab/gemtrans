from torch.optim import Adam


def build(model, config):
    if config.use_ppnet:

        # Lists containing different param groups
        transformer_params = []
        add_on_layers_params = []
        prototype_vectors_params = []
        last_layer_params = []

        # Add the model params to their corresponding lists
        for name, param in model.named_parameters():
            if "add_on_layers" in name:
                add_on_layers_params.append(param)
            elif "prototype_vectors" in name:
                prototype_vectors_params.append(param)
            elif "last_layer" in name:
                last_layer_params.append(param)
            else:
                transformer_params.append(param)

        # Give each param group its optimizer config
        opt_params = [
            {
                "params": add_on_layers_params,
                "lr": config.add_on_layers_lr,
                "weight_decay": 1e-3,
            },
            {"params": prototype_vectors_params, "lr": config.prototype_vectors_lr},
            {"params": last_layer_params, "lr": config.last_layer_lr},
            {
                "params": transformer_params,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            },
        ]

        # define the optimizer
        optimizer = Adam(opt_params, lr=config.lr, weight_decay=config.weight_decay)

    else:
        optimizer = Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    return optimizer

import torch.nn as nn
from src.core.criterion import LvAttnLoss, FrameAttnLoss


def build(config, mode):
    criterion = dict()

    criterion["regression"] = nn.MSELoss()
    criterion["aux_classification"] = nn.CrossEntropyLoss()
    criterion["classification"] = nn.CrossEntropyLoss()
    criterion["spatial_location"] = LvAttnLoss(
        config["frame_size"], config["patches"][0], config["n_sampled_frames"]
    )
    criterion["temporal_location"] = FrameAttnLoss()

    return criterion

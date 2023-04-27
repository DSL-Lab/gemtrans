import os
from torch import distributed as dist
import torch
import numpy as np
import random


###################################### Sanity Check Code ######################################
def general_config_check(config):
    assert config.train.mode in [
        "ef",
        "as",
        "pretrain",
    ], "Mode must be one of ef|as|pretrain"


def check_config_for_criterion(config):
    # Patch and frame attention supervision only enabled for ef mode
    if config.train.mode != "ef":
        assert (
            config.train.criterion.frame_lambda == 0
        ), "ED/ES frame attention supervision only supported for ef mode"

        assert (
            config.train.criterion.attn_lambda == 0
        ), "Patch attention supervision only supported for ef mode"

        assert (
            config.train.criterion.classification_lambda == 0
        ), "Categorized classification only supported for ef mode"


def check_config_for_dataset(config):
    # Only the following datasets are supported
    assert config.data.name in [
        "echonet",
        "as",
        "biplane",
        "kinetics",
    ], "dataset name must be one of as|echonet|biplane|kinetics"

    # Ensure dataset path exists
    assert os.path.exists(config.data.dataset_path), "Dataset path does not exist."


def check_config_checkpoints(config, train):
    # Make sure a model checkpoint is specified if in test mode
    if not train:
        assert (
            config.model.checkpoint_path is not None
        ), "Must specify a model checkpoint for inference."

    # If specified, check the checkpoint file actually exists
    if config.model.checkpoint_path:
        assert os.path.isfile(
            config.model.checkpoint_path
        ), "Model checkpoint not found."


def check_config_for_transformers(config):
    assert (
        config.model.spatial_hidden_size % config.model.spatial_num_heads == 0
    ), "Transformer hidden size must be divisible by the number of heads"

    assert (
        config.model.temporal_hidden_size % config.model.temporal_num_heads == 0
    ), "Transformer hidden size must be divisible by the number of heads"

    assert (
        config.model.vid_hidden_size % config.model.vid_num_heads == 0
    ), "Transformer hidden size must be divisible by the number of heads"

    assert config.model.spatial_aggr_method in [
        "mean",
        "cls",
        "max",
    ], "Aggregation method must be one of cls|mean|max"

    assert config.model.temporal_aggr_method in [
        "mean",
        "cls",
        "max",
    ], "Aggregation method must be one of cls|mean|max"

    assert config.model.vid_aggr_method in [
        "mean",
        "cls",
        "max",
    ], "Aggregation method must be one of cls|mean|max"


def check_config_for_pretrained_vit(config):

    if config.model.pretrained_patch_encoder_path is not None:
        # Make sure pretrained weights exist if needed
        assert os.path.isfile(
            config.model["pretrained_patch_encoder_path"]
        ), "Specified transformer pretrained weights are not found."

        assert (
            config.model.spatial_aggr_method == "cls"
        ), "Aggregation method must be cls for the pretrained VIT."

        assert config.model.patches == [
            16,
            16,
        ], "The patch size must be 16 by 16 for the pretrained VIT."

        assert (
            config.model.spatial_hidden_size == 768
        ), "The hidden dim must be 768 for the pretrained VIT."

        assert (
            config.model.spatial_mlp_dim == 3072
        ), "The MLP hidden dim must be 3072 for the pretrained VIT."

        assert (
            config.model.spatial_num_layers == 12
        ), "Number of layers must be 12 for the pretrained VIT."

        assert (
            config.model.spatial_num_heads == 12
        ), "Number of heads must be 12 for the pretrained VIT."

        assert (
            config.data.frame_size == 224
        ), "The frame size must be 224 for the pretrained VIT."


def sanity_checks(config, train):

    # General config checks
    general_config_check(config=config)

    # Check criterion config
    check_config_for_criterion(config=config)

    # Check dataset config
    check_config_for_dataset(config=config)

    # Make sure checkpoints are properly specified and exist
    check_config_checkpoints(config=config, train=train)

    # If pretrained ViT is used, make sure the config matches the requirements
    check_config_for_pretrained_vit(config=config)

    # Hidden dimension of transformers must be divisible by number of heads
    check_config_for_transformers(config=config)


################################### Other Helper Functions ######################################
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
    Code from: https://github.com/lukemelas/PyTorch-Pretrained-ViT
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from src.core.transformer import Transformer
from src.core.vit.vit_utils import load_pretrained_weights, as_tuple
from src.core.vit.vit_configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        load_repr_layer: bool = True,
        aggr_method: str = "cls",
        positional_embedding: str = "1d",
        in_channels: int = 3,
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        last_layer_attn=False,
        weights_path="./pretrained_models/B_16.pth",
        return_full_attn=False,
    ):
        super().__init__()

        # Configuration
        assert name in PRETRAINED_MODELS.keys(), "name should be in: " + ", ".join(
            PRETRAINED_MODELS.keys()
        )

        self.image_size = image_size

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw
        self.seq_len = seq_len

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw)
        )

        # Class token
        if aggr_method == "cls":
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == "1d":
            self.positional_embedding = PositionalEmbedding1D(self.seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate,
            aggr_method=aggr_method,
            last_layer_attn=last_layer_attn,
            return_full_attn=return_full_attn,
        )

        # Classifier head
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Initialize weights
        self.aggr_method = aggr_method
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]["num_classes"]
            pretrained_image_size = PRETRAINED_MODELS[name]["image_size"]
            load_pretrained_weights(
                self,
                weights_path=weights_path,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
                strict=False,
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight
                )  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        # nn.init.constant_(self.fc.weight, 0)
        # nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(
            self.positional_embedding.pos_embedding, std=0.02
        )  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)

        if self.aggr_method == "cls":
            nn.init.constant_(self.class_token, 0)

    def forward(self, x, mask=None):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, "class_token"):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, "positional_embedding"):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x, debug_attn, last_layer_attn = self.transformer(x, mask)  # b,gh*gw+1,d

        x = self.norm(x)  # b,d

        return x, self.positional_embedding.pos_embedding, debug_attn, last_layer_attn

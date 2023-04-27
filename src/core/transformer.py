"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        ff_dim,
        dropout,
        last_layer_attn=False,
        aggr_method="cls",
        return_full_attn=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.last_layer_attn = last_layer_attn
        self.aggr_method = aggr_method
        self.return_full_attn = return_full_attn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        all_debug_attn = None

        for block in self.blocks:
            x = block(x, mask)

            with torch.no_grad():
                if not self.return_full_attn:
                    if all_debug_attn is None:
                        all_debug_attn = torch.mean(
                            block.attn.scores[0], dim=0
                        ).unsqueeze(0)
                    else:
                        all_debug_attn = torch.cat(
                            [
                                all_debug_attn,
                                torch.mean(block.attn.scores[0], dim=0).unsqueeze(0),
                            ],
                            dim=0,
                        )
                else:
                    if all_debug_attn is None:
                        all_debug_attn = torch.mean(block.attn.scores, dim=1).unsqueeze(
                            1
                        )
                    else:
                        all_debug_attn = torch.cat(
                            [
                                all_debug_attn,
                                torch.mean(block.attn.scores, dim=1).unsqueeze(1),
                            ],
                            dim=1,
                        )

        last_layer_attn = None
        if self.last_layer_attn:
            starting_idx = 1 if self.aggr_method == "cls" else 0
            last_layer_attn = torch.mean(self.blocks[-1].attn.scores, dim=1)[
                :, 0, starting_idx:
            ]

        return x, all_debug_attn, last_layer_attn

    def attn_rollout(self, all_attn, discard_ratio=0.9, head_fusion='mean'):
        result = torch.eye(all_attn[0].shape[-1]).unsqueeze(dim=0).repeat(all_attn[0].shape[0], 1, 1).cuda()
        for attn in all_attn:
            # attn : (batch_size, head_num, 196, 196)
            if head_fusion == "mean":
                attn_fused = attn.mean(axis=1)
            elif head_fusion == "max":
                attn_fused = attn.max(axis=1)[0]    # (batch_size, 196, 196)
            elif head_fusion == "min":
                attn_fused = attn.min(axis=1)[0]

            flat = attn_fused.view(attn_fused.shape[0], -1) # (batch_size, 196 * 196)
            _, indices = flat.topk(int(flat.shape[-1] * discard_ratio), -1, False)
            # flat[indices] = 0
            flat.scatter_(1, indices, 0)

            I = torch.eye(attn_fused.shape[-1]).cuda()
            # a = (attn_fused + 1.0 * I) / 2
            # a = attn_fused  # mark, identity
            identity_w = 0.2
            a = (attn_fused + identity_w * I) / (1. + identity_w)

            a = a / a.sum(dim=-1).unsqueeze(dim=-1)

            result = torch.matmul(a, result)
        return result

    def forward_feature_mask_train_direct(self, cls_embed, x_embed, token_attn=None, reserve_layer_nums=[], mask= None):
        '''
        directly uses the attn rollout as token attn to discard tokens
        cls_embed : (B, 1, dim)
        x_embed : (B, 196, dim)
        '''
        B, patch_num = x_embed.shape[0], x_embed.shape[1]
        layer_ids = [x[0] for x in reserve_layer_nums]
        if mask == None:
            policy = torch.ones(B, 1 + patch_num, device=x_embed.device) # (B, 1 + 196, 1)
        else:
            policy = mask
        x = torch.cat([cls_embed, x_embed], dim=1)  # (B, 1 + 196, dim)
        all_attn = []
        for i, blk in enumerate(self.blocks):
            if i in layer_ids:
                all_attn = all_attn[:i] # (196, 196)
                attn_rollout = self.attn_rollout(all_attn)
                attn_rollout = attn_rollout.detach()    # detach !!!
                cls_token_attn = attn_rollout[:, 0, 1:]

                reserve_token_num = reserve_layer_nums[layer_ids.index(i)][1]
                reserve_token_indice = torch.topk(cls_token_attn, k=reserve_token_num, dim=-1)[1]   # (B, reserve_token_num)
                reserve_token_indice = reserve_token_indice.sort(dim=-1)[0]
                reserve_token_indice += 1   # omit cls token
                policy = torch.cat([torch.ones(B, 1, device=x.device), torch.zeros(B, patch_num, device=x.device)], dim=1)
                policy.scatter_(1, reserve_token_indice,1) # (B, 1 + patch_num)
                policy = policy[:, :]
            x = blk(x, mask= policy)
            attn = blk.attn.scores
            all_attn.append(attn)
        x = self.norm(x)

        return x, (cls_token_attn, None)

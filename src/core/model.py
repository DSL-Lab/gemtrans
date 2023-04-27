import torch
import torch.nn as nn
import numpy as np
from src.core.vit.vit import ViT
from src.core.transformer import Transformer
import math
import yaml
import torch.nn.functional as F


class SpaceTimeFactorizedViViT(nn.Module):
    def __init__(
        self,
        mode,
        patches,
        spatial_dropout_rate,
        spatial_hidden_size,
        spatial_num_layers,
        spatial_mlp_dim,
        spatial_num_heads,
        spatial_aggr_method,
        n_sampled_frames,
        temporal_dropout_rate,
        temporal_hidden_size,
        temporal_num_layers,
        temporal_mlp_dim,
        temporal_num_heads,
        temporal_aggr_method,
        vid_seq_len,
        vid_hidden_size,
        vid_dropout_rate,
        vid_num_layers,
        vid_mlp_dim,
        vid_num_heads,
        vid_aggr_method,
        output_dropout_rate=0.0,
        use_seg_labels=False,
        use_ed_es_locs=False,
        pretrained_patch_encoder_path=None,
        return_full_attn=False,
        use_classification_head=False,
        frame_size=224,
        use_ppnet=False,
    ):

        super(SpaceTimeFactorizedViViT, self).__init__()

        self.temporal_encoder = TemporalEncoder(
            n_sampled_frames,
            patches,
            spatial_hidden_size,
            True,
        )

        n_temporal_tokens = n_sampled_frames
        if temporal_aggr_method == "cls":
            self.temporal_cls_token = nn.Parameter(
                torch.zeros((1, 1, temporal_hidden_size))
            )
            n_temporal_tokens = n_sampled_frames + 1

        n_view_tokens = vid_seq_len
        if vid_aggr_method == "cls":
            self.vid_cls_token = nn.Parameter(torch.zeros((1, 1, vid_hidden_size)))
            n_view_tokens = vid_seq_len + 1

        self.spatial_trans_enc = ViT(
            "B_16",
            patches=patches[0],
            dim=spatial_hidden_size,
            ff_dim=spatial_mlp_dim,
            num_heads=spatial_num_heads,
            num_layers=spatial_num_layers,
            dropout_rate=spatial_dropout_rate,
            weights_path=pretrained_patch_encoder_path,
            last_layer_attn=use_seg_labels,
            pretrained=True if pretrained_patch_encoder_path is not None else False,
            aggr_method=spatial_aggr_method,
            return_full_attn=return_full_attn,
            image_size=frame_size,
        )

        # If ViT's hidden dim doesn't match that of temporal transformer, we need to project
        self.ste_to_tte = None
        if spatial_hidden_size != temporal_hidden_size:
            self.ste_to_tte = nn.Linear(spatial_hidden_size, temporal_hidden_size)

        self.temporal_trans_enc = Encoder(
            num_layers=temporal_num_layers,
            mlp_dim=temporal_mlp_dim,
            dropout_rate=temporal_dropout_rate,
            hidden_size=temporal_hidden_size,
            num_heads=temporal_num_heads,
            seq_len=n_temporal_tokens,
            aggr_method=temporal_aggr_method,
            last_layer_attn=use_ed_es_locs,
            return_full_attn=return_full_attn,
        )

        self.tte_to_vte = None
        if temporal_hidden_size != vid_hidden_size:
            self.tte_to_vte = nn.Linear(temporal_hidden_size, vid_hidden_size)

        self.vid_trans_enc = Encoder(
            num_layers=vid_num_layers,
            mlp_dim=vid_mlp_dim,
            dropout_rate=vid_dropout_rate,
            hidden_size=vid_hidden_size,
            num_heads=vid_num_heads,
            seq_len=n_view_tokens,
            aggr_method=vid_aggr_method,
            last_layer_attn=False,
            return_full_attn=return_full_attn,
        )

        self.use_ppnet = use_ppnet
        if use_ppnet:
            img_size = 224
            prototype_shape = [32, 192, 1, 1]
            num_classes = 4
            reserve_layers = [11]
            reserve_token_nums = [81]
            proto_layer_rf_info = [14, 16, 16, 8.0]
            use_global = True
            use_ppc_loss = True
            ppc_cov_thresh = 1
            ppc_mean_thresh = 2
            global_coe = 0.5
            global_proto_per_class = 4
            prototype_activation_function = "log"
            add_on_layers_type = "regular"
            self.prototype_layer = PPNet(
                features=self.spatial_trans_enc,
                img_size=img_size,
                prototype_shape=prototype_shape,
                proto_layer_rf_info=proto_layer_rf_info,
                num_classes=num_classes,
                reserve_layers=reserve_layers,
                reserve_token_nums=reserve_token_nums,
                use_global=use_global,
                use_ppc_loss=use_ppc_loss,
                ppc_cov_thresh=ppc_cov_thresh,
                ppc_mean_thresh=ppc_mean_thresh,
                global_coe=global_coe,
                global_proto_per_class=global_proto_per_class,
                init_weights=True,
                prototype_activation_function=prototype_activation_function,
                add_on_layers_type=add_on_layers_type,
            )

        if mode == "ef":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 1),
                nn.Sigmoid(),
            )
            # set bias to 0.552
            self.output_mlp[-2].bias.data[0] = 0.552

            self.class_output_mlp = None
            if use_classification_head:
                self.class_output_mlp = nn.Sequential(
                    nn.Linear(
                        in_features=vid_hidden_size, out_features=vid_hidden_size // 2
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=output_dropout_rate),
                    nn.Linear(in_features=vid_hidden_size // 2, out_features=4),
                )

        elif mode == "as":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 4),
            )

            self.class_output_mlp = None
        elif mode == "pretrain":
            self.output_mlp = nn.Sequential(
                nn.Linear(vid_hidden_size, vid_hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 2, vid_hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(vid_hidden_size // 4, 400),
            )

            self.class_output_mlp = None
        else:
            self.output_mlp = None
            self.class_output_mlp = None

        self.mode = mode
        self.spatial_aggr_method = spatial_aggr_method
        self.temporal_aggr_method = temporal_aggr_method
        self.vid_aggr_method = vid_aggr_method
        self.use_ed_es_locs = use_ed_es_locs
        self.use_pretrained_patch_encoder = pretrained_patch_encoder_path is not None
        self.n_sampled_frames = n_sampled_frames
        self.return_full_attn = return_full_attn

    def forward(self, data_dict):

        x = data_dict["vid"]
        mask = data_dict["mask"]
        ed_frames = data_dict["ed_frame"]
        ed_valid = data_dict["ed_valid"]
        es_frames = data_dict["es_frame"]
        es_valid = data_dict["es_valid"]
        label = data_dict["label"] if data_dict["label"].dtype is torch.long else None

        num_frames = x.shape[2]
        x, mask = self.temporal_encoder(x, mask)

        # Proto-related variables
        logits = None
        auxi_item = None
        ppc_loss = None

        # Return sampled video for visualization purposes
        with torch.no_grad():
            sampled_vid = x.detach() if self.return_full_attn else None

        # The frames are subsampled now, modify ED/ES locations accordingly
        if self.use_ed_es_locs and ed_valid is not None:
            ed_valid, es_valid, ed_frames, es_frames = sample_ed_es(
                ed_frames,
                ed_valid,
                es_frames,
                es_valid,
                num_frames,
                self.n_sampled_frames,
            )

        bs, nvid, t, h, w, c = x.shape

        # Change to shape required for STE
        x = x.permute(0, 1, 2, 5, 3, 4)
        x = x.contiguous().view(
            x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
        )

        # Prepare patch wise mask from the frame level mask
        patch_mask = (
            mask.contiguous()
            .view(bs * nvid * t, 1)
            .repeat(1, self.spatial_trans_enc.seq_len)
        )

        (
            x,
            patch_pos_embed,
            patch_attn,
            last_layer_patch_attn,
        ) = self.spatial_trans_enc(x, patch_mask)

        if last_layer_patch_attn is not None:
            last_layer_patch_attn = last_layer_patch_attn.contiguous().view(
                bs, last_layer_patch_attn.shape[0] // bs, -1
            )

        if self.use_ppnet:
            logits, auxi_item = self.prototype_layer(
                x, patch_mask
            )  # x.shape(B,197,768)
            if label is not None:
                total_proto_act, cls_attn_rollout, original_fea_len = (
                    auxi_item[2],
                    auxi_item[3],
                    auxi_item[4],
                )
                ppc_cov_loss, ppc_mean_loss = self.prototype_layer.get_PPC_loss(
                    total_proto_act,
                    cls_attn_rollout,
                    original_fea_len,
                    bs * nvid,
                    label,
                )
                ppc_cov_loss = 0.1 * ppc_cov_loss
                ppc_mean_loss = 0.5 * ppc_mean_loss
                ppc_loss = ppc_cov_loss + ppc_mean_loss
            logits = torch.reshape(logits, (bs * nvid, t, 4))
            logits = torch.mean(logits, dim=1)

        # Aggregate tokens
        if self.spatial_aggr_method == "cls":
            x = x[:, 0]
        elif self.spatial_aggr_method == "mean":
            sums = torch.sum(patch_mask, dim=1, keepdim=True)
            x = torch.sum(x * patch_mask.unsqueeze(-1), dim=1) / sums
        elif self.spatial_aggr_method == "max":
            x = x.max(dim=1).values

        # Change to shape required for TTE
        x = x.contiguous().view(bs * nvid, x.shape[0] // (bs * nvid), x.shape[-1])

        if self.ste_to_tte is not None:
            x = self.ste_to_tte(x)

        n, _, c = x.shape

        frame_mask = mask.contiguous().view(n, -1)

        if self.temporal_aggr_method == "cls":
            temporal_cls_tokens = torch.tile(self.temporal_cls_token, [n, 1, 1])
            x = torch.cat((temporal_cls_tokens, x), dim=1)

            frame_mask = torch.cat(
                (torch.ones((n, 1), dtype=torch.bool, device=mask.device), frame_mask),
                dim=1,
            )

        # Temporal encoder
        x, frame_pos_embed, frame_attn, last_layer_frame_attn = self.temporal_trans_enc(
            x, frame_mask
        )

        if self.temporal_aggr_method == "cls":
            x = x[:, 0]
        elif self.temporal_aggr_method == "mean":
            sums = torch.sum(frame_mask, dim=1, keepdim=True)
            x = torch.sum(x * frame_mask.unsqueeze(-1), dim=1) / sums
        elif self.temporal_aggr_method == "max":
            x = x.max(dim=1).values

        # Change to shape needed for VTE
        x = x.contiguous().view(bs, x.shape[0] // bs, x.shape[-1])

        if self.tte_to_vte is not None:
            x = self.tte_to_vte(x)

        # Add class tokens
        n, _, c = x.shape
        vid_mask = torch.any(mask, dim=2)
        if self.vid_aggr_method == "cls":
            vid_cls_tokens = torch.tile(self.vid_cls_token, [n, 1, 1])
            x = torch.cat((vid_cls_tokens, x), dim=1)
            vid_mask = torch.cat(
                (torch.ones((n, 1), dtype=torch.bool, device=mask.device), vid_mask),
                dim=1,
            )

        # Video encoder
        # x[torch.bitwise_not(vid_mask)] = 0
        x, vid_pos_embed, vid_attn, _ = self.vid_trans_enc(x, vid_mask)

        if self.vid_aggr_method == "cls":
            x = x[:, 0]
        elif self.vid_aggr_method == "mean":
            sums = torch.sum(vid_mask, dim=1, keepdim=True)
            x = torch.sum(x * vid_mask.unsqueeze(-1), dim=1) / sums
        elif self.vid_aggr_method == "max":
            x = torch.nan_to_num(x)
            x = x.max(dim=1).values

        x_class = None
        if self.class_output_mlp is not None:
            x_class = self.class_output_mlp(x)

        if self.output_mlp is not None:
            x = self.output_mlp(x)

        return {
            "x": x,
            "patch_pos_embed": patch_pos_embed,
            "frame_pos_embed": frame_pos_embed,
            "vid_pos_embed": vid_pos_embed,
            "patch_attn": patch_attn,
            "frame_attn": frame_attn,
            "vid_attn": vid_attn,
            "last_layer_patch_attn": last_layer_patch_attn,
            "last_layer_frame_attn": last_layer_frame_attn,
            "ed_valid": ed_valid,
            "es_valid": es_valid,
            "ed_frames": ed_frames,
            "es_frames": es_frames,
            "sampled_vid": sampled_vid,
            "x_class": x_class,
            "logits": logits,
            "ppc_loss": ppc_loss,
        }

    def extract_windows(
        self, x, bs, hidden_dim, n_patches, n_frames, windows_per_frame
    ):

        x = (
            x.permute(0, 2, 1)
            .contiguous()
            .view(
                bs * n_frames,
                hidden_dim,
                int(math.sqrt(n_patches)),
                int(math.sqrt(n_patches)),
            )
        )

        # Extract windows
        x = (
            x.unfold(1, hidden_dim, hidden_dim)
            .unfold(2, self.cross_attn_window, self.cross_attn_window)
            .unfold(3, self.cross_attn_window, self.cross_attn_window)
        )

        unfold_shape = x.size()

        x = x.contiguous().view(
            bs * n_frames,
            -1,
            hidden_dim,
            self.cross_attn_window,
            self.cross_attn_window,
        )

        x = x.contiguous().view(
            bs, n_frames, windows_per_frame, hidden_dim, self.cross_attn_window**2
        )
        x = x.permute(0, 2, 1, 4, 3)
        x = x.contiguous().view(
            bs * windows_per_frame, n_frames * self.cross_attn_window**2, hidden_dim
        )

        return x, unfold_shape

    def recons_frame(
        self, x, unfold_shape, bs, windows_per_frame, n_patches, n_frames, hidden_dim
    ):
        x = x.contiguous().view(
            bs, windows_per_frame, n_frames, self.cross_attn_window**2, hidden_dim
        )
        x = x.permute(0, 2, 1, 4, 3)
        x = x.contiguous().view(
            bs * n_frames,
            windows_per_frame,
            hidden_dim,
            self.cross_attn_window,
            self.cross_attn_window,
        )

        x = x.contiguous().view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(-1, output_c, output_h, output_w)

        x = x.contiguous().view(bs * n_frames, hidden_dim, n_patches).permute(0, 2, 1)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        mlp_dim,
        dropout_rate,
        hidden_size,
        num_heads,
        seq_len,
        aggr_method="cls",
        last_layer_attn=False,
        return_full_attn=False,
    ):
        super(Encoder, self).__init__()

        self.positional_embedder = PositionEmbs(seq_len, hidden_size, dropout_rate)

        # self.pre_logits = nn.Linear(hidden_size, repr_dim)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.transformer = Transformer(
            num_layers=num_layers,
            dim=hidden_size,
            num_heads=num_heads,
            ff_dim=mlp_dim,
            dropout=dropout_rate,
            last_layer_attn=last_layer_attn,
            aggr_method=aggr_method,
            return_full_attn=return_full_attn,
        )

        # Initialize weights
        self.init_weights()

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
        nn.init.normal_(
            self.positional_embedder.pos_emb, std=0.02
        )  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)

    def forward(self, x, mask):
        x = self.positional_embedder(x)

        x, debug_attn, last_layer_attn = self.transformer(x, mask)

        x = self.layer_norm(x)  # b,d

        return x, self.positional_embedder.pos_emb.data, debug_attn, last_layer_attn


class PositionEmbs(nn.Module):
    def __init__(self, seq_len, hidden_size, dropout_rate):
        super(PositionEmbs, self).__init__()

        self.pos_emb = nn.Parameter(
            torch.zeros(1, seq_len, hidden_size), requires_grad=True
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.pos_emb
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    def __init__(
        self, n_sampled_frames, patches, hidden_size, use_pretrained_patch_encoder
    ):

        super(TemporalEncoder, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=(patches[0], patches[1]),
            stride=(patches[0], patches[1]),
        )

        self.n_sampled_frames = n_sampled_frames
        self.patches = patches
        self.hidden_size = hidden_size
        self.use_pretrained_patch_encoder = use_pretrained_patch_encoder

    def forward(self, x, mask):

        # Choose frames
        x = sample_frames_uniformly(x, self.n_sampled_frames)
        mask = sample_frames_uniformly(mask, self.n_sampled_frames)
        bs, n, ts, in_h, in_w, c = x.shape

        if self.use_pretrained_patch_encoder:
            x = x.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], 3)
        else:
            # Change to channels-second format
            x = x.permute(0, 5, 1, 2, 3, 4)

            # Reshape to an elongated frame
            x = x.contiguous().view(bs, c, n * ts * in_h, in_w)

            # Embed the patches
            x = self.conv(x)
            bs, c, nth, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.contiguous().view(bs, n, ts, -1, w, c)

        return x, mask


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, in_dim, out_dim, hidden_size, dropout_rate):
        super(MlpBlock, self).__init__()

        self.dense1 = nn.Linear(in_dim, hidden_size)
        self.activation_func = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


def sample_frames_uniformly(x, n_sampled_frames):
    num_frames = x.shape[2]
    temporal_indices = np.linspace(
        start=0, stop=num_frames, num=n_sampled_frames, endpoint=False, dtype=np.int32
    )
    return x[:, :, temporal_indices]


def sample_ed_es(
    ed_frames, ed_valid, es_frames, es_valid, num_frames, n_sampled_frames
):
    ed_valid = ed_valid.flatten()
    es_valid = es_valid.flatten()
    ed_frames = ed_frames.flatten()
    es_frames = es_frames.flatten()

    temporal_indices = np.linspace(
        start=0, stop=num_frames, num=n_sampled_frames, endpoint=False, dtype=np.int32
    )

    for i in range(ed_valid.shape[0]):
        if ed_valid[i]:
            if ed_frames[i].item() not in temporal_indices:
                ed_valid[i] = False
            else:
                ed_frames[i] = np.where(temporal_indices == ed_frames[i].item())[
                    0
                ].item()

        if es_valid[i]:
            if es_frames[i].item() not in temporal_indices:
                es_valid[i] = False
            else:
                es_frames[i] = np.where(temporal_indices == es_frames[i].item())[
                    0
                ].item()

    return ed_valid, es_valid, ed_frames, es_frames


class PPNet(nn.Module):
    def __init__(
        self,
        features,
        img_size,
        prototype_shape,
        proto_layer_rf_info,
        num_classes,
        reserve_layers=[],
        reserve_token_nums=[],
        use_global=False,
        use_ppc_loss=False,
        ppc_cov_thresh=1.0,
        ppc_mean_thresh=2,
        global_coe=0.3,
        global_proto_per_class=10,
        init_weights=True,
        prototype_activation_function="log",
        add_on_layers_type="bottleneck",
    ):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.reserve_layers = reserve_layers
        self.reserve_token_nums = reserve_token_nums
        self.use_global = use_global
        self.use_ppc_loss = use_ppc_loss
        self.ppc_cov_thresh = ppc_cov_thresh
        self.ppc_mean_thresh = ppc_mean_thresh
        self.global_coe = global_coe
        self.global_proto_per_class = global_proto_per_class
        self.epsilon = 1e-4
        self.reserve_layer_nums = list(
            zip(self.reserve_layers, self.reserve_token_nums)
        )

        self.num_prototypes_global = self.num_classes * self.global_proto_per_class
        self.prototype_shape_global = [
            self.num_prototypes_global
        ] + self.prototype_shape[1:]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        """
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        """
        assert self.num_prototypes % self.num_classes == 0
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes, self.num_classes
        )
        self.prototype_class_identity_global = torch.zeros(
            self.num_prototypes_global, self.num_classes
        )

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        num_prototypes_per_class_global = self.num_prototypes_global // self.num_classes
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[
                j, j // num_prototypes_per_class_global
            ] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        # Input channel size
        first_add_on_layer_in_channels = [
            i for i in features.modules() if isinstance(i, nn.Linear)
        ][-1].out_features

        self.num_patches = 16

        if add_on_layers_type == "bottleneck":
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (
                len(add_on_layers) == 0
            ):
                current_out_channels = max(
                    self.prototype_shape[1], (current_in_channels // 2)
                )
                add_on_layers.append(
                    nn.Conv2d(
                        in_channels=current_in_channels,
                        out_channels=current_out_channels,
                        kernel_size=1,
                    )
                )
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(
                    nn.Conv2d(
                        in_channels=current_out_channels,
                        out_channels=current_out_channels,
                        kernel_size=1,
                    )
                )
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert current_out_channels == self.prototype_shape[1]
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=first_add_on_layer_in_channels,
                    out_channels=self.prototype_shape[1],
                    kernel_size=1,
                ),
                nn.Sigmoid(),
            )

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )
        if self.use_global:
            self.prototype_vectors_global = nn.Parameter(
                torch.rand(self.prototype_shape_global), requires_grad=True
            )

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(
            self.num_prototypes, self.num_classes, bias=False
        )  # do not use bias
        self.last_layer_global = nn.Linear(
            self.num_prototypes_global, self.num_classes, bias=False
        )  # do not use bias
        self.last_layer.weight.requires_grad = False
        self.last_layer_global.weight.requires_grad = False

        self.all_attn_mask = None
        self.teacher_model = None

        self.scale = self.prototype_shape[1] ** -0.5

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x, reserve_layer_nums=[], mask=None):
        """
        the feature input to prototype layer
        """
        batch_size = x.shape[0]
        cls_embed, x_embed = x[:, :1], x[:, 1:]  # (B, 1, dim), (B, 196, dim)
        fea_size, dim = int(x_embed.shape[1] ** (1 / 2)), x_embed.shape[-1]

        token_attn = None
        x, (
            cls_token_attn,
            _,
        ) = self.features.transformer.forward_feature_mask_train_direct(
            cls_embed, x_embed, token_attn, reserve_layer_nums, mask
        )
        final_reserve_num = reserve_layer_nums[-1][1]
        final_reserve_indices = torch.topk(cls_token_attn, k=final_reserve_num, dim=-1)[
            1
        ]  # (B, final_reserve_num)
        final_reserve_indices = final_reserve_indices.sort(dim=-1)[0]
        final_reserve_indices = final_reserve_indices[:, :, None].repeat(
            1, 1, dim
        )  # (B, final_reserve_num, dim)

        cls_tokens, img_tokens = x[:, :1], x[:, 1:]  # (B, 1, dim), (B, 196, dim)
        img_tokens = torch.gather(
            img_tokens, 1, final_reserve_indices
        )  # (B, final_reserve_num, dim)

        B, dim, fea_len = img_tokens.shape[0], img_tokens.shape[2], img_tokens.shape[1]
        fea_width, fea_height = int(fea_len ** (1 / 2)), int(fea_len ** (1 / 2))
        cls_tokens = cls_tokens.permute(0, 2, 1).reshape(
            B, dim, 1, 1
        )  # (batch_size, dim, 1, 1)
        img_tokens = img_tokens.permute(0, 2, 1).reshape(
            B, dim, fea_height, fea_width
        )  # (batch_size, dim, fea_size, fea_size)

        cls_tokens = self.add_on_layers(cls_tokens)
        img_tokens = self.add_on_layers(img_tokens)

        return (cls_tokens, img_tokens), (token_attn, cls_token_attn, None)

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        """
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        """
        input2 = input**2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter**2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = (
            -2 * weighted_inner_product + filter_weighted_norm2_reshape
        )
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution_single(self, x, prototype_vectors):
        temp_ones = torch.ones(prototype_vectors.shape).cuda()

        x2 = x**2
        x2_patch_sum = F.conv2d(input=x2, weight=temp_ones)

        p2 = prototype_vectors**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = -2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x, reserve_layer_nums=[], mask=None):
        """
        x is the raw input
        """
        if self.use_global:
            # (cls_tokens, img_tokens), auxi_item = self.conv_features(x, reserve_layer_nums)
            (cls_tokens, img_tokens), auxi_item = self.conv_features(
                x, reserve_layer_nums, mask
            )
            return (cls_tokens, img_tokens), auxi_item

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def get_activations(self, tokens, prototype_vectors):
        batch_size, num_prototypes = tokens.shape[0], prototype_vectors.shape[0]
        distances = self._l2_convolution_single(tokens, prototype_vectors)
        activations = self.distance_2_similarity(distances)  # (B, 2000, 1, 1)
        total_proto_act = activations
        fea_size = activations.shape[-1]
        if fea_size > 1:
            activations = F.max_pool2d(
                activations, kernel_size=(fea_size, fea_size)
            )  # (B, 2000, 1, 1)
        activations = activations.reshape(batch_size, num_prototypes)
        if self.use_global:
            return activations, (distances, total_proto_act)
        return activations

    def batch_cov(self, points, weights):
        B, N, D = points.size()  # weights : (B, N)
        weights = weights / weights.sum(dim=-1, keepdim=True) * N  # (B, N)
        mean = (points * weights[:, :, None]).mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        prods = prods * weights[:, :, None, None]
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return mean, bcov  # (B, D, D)

    def get_PPC_loss(
        self, total_proto_act, cls_attn_rollout, original_fea_len, batch, label
    ):
        total_proto_act = total_proto_act.reshape(
            batch,
            total_proto_act.shape[0] // batch,
            total_proto_act.shape[1],
            total_proto_act.shape[2],
            total_proto_act.shape[3],
        )
        cls_attn_rollout = cls_attn_rollout.reshape(
            batch, cls_attn_rollout.shape[0] // batch, cls_attn_rollout.shape[1]
        )
        num_frames = total_proto_act.shape[1]
        batch_size, original_fea_size = total_proto_act.shape[0], int(
            original_fea_len ** (1 / 2)
        )
        proto_per_class = self.num_prototypes_per_class
        discrete_values = torch.FloatTensor(
            [[x, y] for x in range(original_fea_size) for y in range(original_fea_size)]
        ).cuda()  # (196, 2)
        discrete_values = discrete_values[None, :, :].repeat(
            batch_size * num_frames * proto_per_class, 1, 1
        )  # (B * F * 8,196, 2)
        discrete_weights = torch.zeros(
            batch_size, num_frames, proto_per_class, original_fea_len
        ).cuda()  # (B, F, 8, 196)
        total_proto_act = total_proto_act.flatten(start_dim=3)  # (B, F, 32, 81)
        final_token_num = total_proto_act.shape[-1]  # 81
        # select the prototypes corresponding to the label
        proto_indices = (
            (label * proto_per_class)
            .unsqueeze(dim=-1)
            .repeat(1, num_frames, proto_per_class)
        )
        proto_indices += torch.arange(
            proto_per_class
        ).cuda()  # (B, 10), get 10 indices of activation maps of each sample
        proto_indices = proto_indices[:, :, :, None].repeat(1, 1, 1, final_token_num)
        total_proto_act = torch.gather(total_proto_act, 1, proto_indices)  # (B, 10, 81)

        reserve_token_indices = torch.topk(cls_attn_rollout, k=final_token_num, dim=-1)[
            1
        ]  # (B, 81)
        reserve_token_indices = reserve_token_indices.sort(dim=-1)[0]
        reserve_token_indices = reserve_token_indices[:, :, None, :].repeat(
            1, 1, proto_per_class, 1
        )  # (B, F, 8, 81)
        discrete_weights.scatter_(
            3, reserve_token_indices, total_proto_act
        )  # (B, F, 8, 196)
        discrete_weights = discrete_weights.reshape(
            batch_size * proto_per_class * num_frames, -1
        )  # (B * F * 8, 196)
        mean_ma, cov_ma = self.batch_cov(
            discrete_values, discrete_weights
        )  # (B * F * 8, 2, 2)
        # cov loss
        ppc_cov_loss = (cov_ma[:, 0, 0] + cov_ma[:, 1, 1]) / 2
        ppc_cov_loss = F.relu(ppc_cov_loss - self.ppc_cov_thresh).mean()
        # mean loss
        mean_ma = mean_ma.reshape(
            batch_size * num_frames, proto_per_class, 2
        )  # (B* F, 10, 2)
        mean_diff = torch.cdist(mean_ma, mean_ma)  # (B, 10, 10)
        mean_mask = 1.0 - torch.eye(proto_per_class).cuda()  # (10, 10)
        ppc_mean_loss = F.relu((self.ppc_mean_thresh - mean_diff) * mean_mask).mean()

        return ppc_cov_loss, ppc_mean_loss

    def forward(self, x, mask=None):
        reserve_layer_nums = self.reserve_layer_nums
        if not self.training:
            if self.use_global:
                (cls_tokens, img_tokens), (
                    token_attn,
                    cls_token_attn,
                    _,
                ) = self.prototype_distances(x, reserve_layer_nums, mask)
                global_activations, _ = self.get_activations(
                    cls_tokens, self.prototype_vectors_global
                )
                local_activations, (distances, _) = self.get_activations(
                    img_tokens, self.prototype_vectors
                )

                logits_global = self.last_layer_global(global_activations)
                logits_local = self.last_layer(local_activations)
                logits = (
                    self.global_coe * logits_global
                    + (1.0 - self.global_coe) * logits_local
                )
                return logits, (cls_token_attn, distances, logits_global, logits_local)

        # re-calculate distances
        if self.use_global:
            (cls_tokens, img_tokens), (
                student_token_attn,
                cls_attn_rollout,
                _,
            ) = self.prototype_distances(x, reserve_layer_nums, mask)
            cls_attn_rollout = cls_attn_rollout.detach()  # detach
            # get token attention loss
            batch_size, fea_size, original_fea_size = (
                cls_tokens.shape[0],
                img_tokens.shape[-1],
                int(cls_attn_rollout.shape[-1] ** (1 / 2)),
            )
            teacher_token_attn = cls_attn_rollout

            global_activations, _ = self.get_activations(
                cls_tokens, self.prototype_vectors_global
            )
            local_activations, (_, total_proto_act) = self.get_activations(
                img_tokens, self.prototype_vectors
            )

            logits_global = self.last_layer_global(global_activations)
            logits_local = self.last_layer(local_activations)
            logits = (
                self.global_coe * logits_global + (1.0 - self.global_coe) * logits_local
            )
        else:
            distances, (student_token_attn, _, _) = self.prototype_distances(
                x, reserve_layer_nums, mask
            )
            # global min pooling
            batch_size, fea_size = distances.shape[0], distances.shape[-1]
            prototype_activations = self.distance_2_similarity(
                distances
            )  # (B, 2000, 9, 9)
            total_proto_act = prototype_activations  # (B, 2000, 9, 9)

            prototype_activations = F.max_pool2d(
                prototype_activations, kernel_size=(fea_size, fea_size)
            )
            prototype_activations = prototype_activations.view(-1, self.num_prototypes)

            logits = self.last_layer(prototype_activations)

        attn_loss = torch.zeros(1, device=logits.device)
        original_fea_len = original_fea_size**2
        return logits, (
            student_token_attn,
            attn_loss,
            total_proto_act,
            cls_attn_rollout,
            original_fea_len,
        )

    def push_forward(self, x):
        """this method is needed for the pushing operation"""
        reserve_layer_nums = self.reserve_layer_nums
        (cls_tokens, img_tokens), (
            token_attn,
            cls_token_attn,
            _,
        ) = self.prototype_distances(x, reserve_layer_nums)
        global_activations, _ = self.get_activations(
            cls_tokens, self.prototype_vectors_global
        )
        local_activations, (distances, proto_acts) = self.get_activations(
            img_tokens, self.prototype_vectors
        )

        return cls_token_attn, proto_acts

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tfeatures: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(
            self.features,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
            self.epsilon,
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

        if hasattr(self, "last_layer_global"):
            positive_one_weights_locations = torch.t(
                self.prototype_class_identity_global
            )
            negative_one_weights_locations = 1 - positive_one_weights_locations

            self.last_layer_global.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations
            )

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

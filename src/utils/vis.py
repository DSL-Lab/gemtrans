from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import SimpleITK as sitk
from torchvision.transforms import ToTensor
import torch

try:
    import wandb
except ImportError:
    pass


def plot_cos_sim(dist, labels, save_dir):
    save_dir = os.path.join(save_dir, "cos_sim_plots")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(labels)):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        idx = np.argsort(dist[i, :])
        ax.bar(np.arange(len(labels)), dist[i, idx], log=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels[idx], rotation=90)
        plt.title("EF = {}".format(labels[i]))

        plt.savefig(os.path.join(save_dir, "embeddings_" + str(i) + ".png"))
        plt.close()


def plot_2d_pos_embds(
    epoch,
    iteration,
    use_wandb,
    save_dir,
    pos_embs,
    num_rows,
    num_cols,
    log_str,
    step_metric,
    mode,
):
    step_name, step_value = step_metric.popitem()

    # Generate the cosine similarity matrix
    pos_embs = pos_embs.reshape(num_rows, num_cols, -1)

    patch_cos_sim_matrices = np.zeros((num_rows, num_cols, num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            for row_in_patch in range(num_rows):
                for col_in_patch in range(num_cols):
                    patch_cos_sim_matrices[
                        row, col, row_in_patch, col_in_patch
                    ] = 1 - distance.cosine(
                        pos_embs[row, col, :], pos_embs[row_in_patch, col_in_patch, :]
                    )

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(64, 64))
    for row in range(num_rows):
        for col in range(num_cols):
            ax[row, col].imshow(patch_cos_sim_matrices[row, col], cmap="hot")

    if use_wandb:
        wandb.log(
            {
                f"{mode}/" + log_str + "_pos_embs": wandb.Image(fig),
                f"{mode}/{step_name}": step_value,
            }
        )
    else:
        plt.savefig(
            os.path.join(
                save_dir,
                "visualizations",
                "patch_level",
                "{}_pos_emb_{}_{}.png".format(mode, epoch, iteration),
            ),
        )

    plt.close("all")


def plot_1d_pos_embds(
    epoch,
    iteration,
    pos_embs,
    seq_len,
    log_str,
    step_metric,
    mode,
    use_wandb,
    save_dir,
):
    step_name, step_value = step_metric.popitem()

    # Generate the cosine similarity matrix
    cos_sim_matrix = np.zeros((seq_len, seq_len))
    for row in range(seq_len):
        for col in range(seq_len):
            cos_sim_matrix[row, col] = 1 - distance.cosine(
                pos_embs[row, :], pos_embs[col, :]
            )

    fig, ax = plt.subplots(1, 1)
    ax.imshow(cos_sim_matrix, cmap="hot")

    if use_wandb:
        wandb.log(
            {
                f"{mode}/" + log_str + "_pos_embs": wandb.Image(fig),
                f"{mode}/{step_name}": step_value,
            }
        )
    else:
        plt.savefig(
            os.path.join(
                save_dir,
                "visualizations",
                "{}_level".format(log_str),
                "{}_{}_pos_emb_{}_{}.png".format(mode, log_str, epoch, iteration),
            ),
        )

    plt.close("all")


def plot_attn_with_img(
    epoch,
    iteration,
    img,
    attn_weights,
    log_str,
    step_metric,
    mode,
    aggr_method,
    use_wandb,
    save_dir,
    test=False,
    frame_attn=None,
    frame_num=0,
    label=None,
    sample_idx=0,
):
    step_name, step_value = step_metric.popitem()

    v, aug_attn_weights = attention_rollout(attn_weights)

    if test:
        v_frame, _ = attention_rollout(frame_attn)

    if aggr_method == "cls":
        grid_size = int(np.sqrt(aug_attn_weights.shape[-1]))
        mask = v[0, 1:].reshape(grid_size, grid_size)

        mask = cv2.resize(mask / mask.max(), img.shape)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(mask, cmap="jet")

        img_3c = cv2.normalize(
            np.stack([img, img, img], axis=2), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        mask_norm = cv2.applyColorMap(
            cv2.normalize(mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U),
            cv2.COLORMAP_JET,
        )
        result = cv2.addWeighted(
            img_3c, 0.4, cv2.cvtColor(mask_norm, cv2.COLOR_RGB2BGR), 0.6, 0
        )
        ax[2].imshow(result, cmap="jet")

        if test:

            # Add a bar indicating the attention given to each frame
            result[
                0:10,
                0 : int(
                    img.shape[1] * v_frame[0, 1 + frame_num] / np.max(v_frame[0, 1:])
                ),
            ] = 1.0

            save_path = os.path.join(
                save_dir,
                "visualizations",
                "test_vis",
                "sample_{}_label_{}".format(sample_idx, label),
            )
            os.makedirs(save_path, exist_ok=True)
            plt.imsave(
                os.path.join(save_path, "frame_{}.png".format(frame_num)),
                result,
                cmap="gray",
            )
    else:
        fig, ax = plt.subplots(1)
        ax.imshow(v, cmap="hot")

    if use_wandb:
        wandb.log(
            {
                f"{mode}/" + log_str + "_attn": wandb.Image(fig),
                f"{mode}/{step_name}": step_value,
            }
        )
    else:
        plt.savefig(
            os.path.join(
                save_dir,
                "visualizations",
                "patch_level",
                "{}_patch_attn_{}_{}.png".format(mode, epoch, iteration),
            ),
        )

    plt.close("all")


def plot_all_pos_embeds(
    epoch,
    iteration,
    use_wandb,
    save_dir,
    patch_pos,
    frame_pos,
    vid_pos,
    config,
    step,
    mode,
):
    starting_idx = 1 if config["spatial_aggr_method"] == "cls" else 0
    if patch_pos is not None:
        plot_2d_pos_embds(
            epoch=epoch,
            iteration=iteration,
            use_wandb=use_wandb,
            save_dir=save_dir,
            pos_embs=patch_pos.cpu().numpy()[0][starting_idx:],
            num_rows=config["frame_size"] // config["patches"][0],
            num_cols=config["frame_size"] // config["patches"][1],
            log_str="patch",
            step_metric={"step": step},
            mode=mode,
        )

    starting_idx = 1 if config["temporal_aggr_method"] == "cls" else 0
    if frame_pos is not None:
        plot_1d_pos_embds(
            epoch=epoch,
            iteration=iteration,
            use_wandb=use_wandb,
            save_dir=save_dir,
            pos_embs=frame_pos.cpu().numpy()[0][starting_idx:],
            seq_len=config["n_sampled_frames"],
            log_str="frame",
            step_metric={"step": step},
            mode=mode,
        )

    starting_idx = 1 if config["vid_aggr_method"] == "cls" else 0
    if vid_pos is not None:
        plot_1d_pos_embds(
            epoch=epoch,
            iteration=iteration,
            use_wandb=use_wandb,
            save_dir=save_dir,
            pos_embs=vid_pos.cpu().numpy()[0][starting_idx:],
            seq_len=config["vid_seq_len"],
            log_str="vid",
            step_metric={"step": step},
            mode=mode,
        )


def plot_all_attn_maps(
    epoch,
    iteration,
    use_wandb,
    save_dir,
    vids,
    patch_attn,
    frame_attn,
    vid_attn,
    step,
    mode,
    config,
    ed_frame=None,
    ed_valid=None,
    es_frame=None,
    es_valid=None,
    sample_idx=0,
    label=None,
    num_frames=32,
    test=False,
):
    # Plot the attention on the patch level superimposed on the image
    if patch_attn is not None:
        if test:
            for idx in range(num_frames):
                plot_attn_with_img(
                    epoch=epoch,
                    iteration=iteration,
                    use_wandb=use_wandb,
                    save_dir=save_dir,
                    img=vids[0, idx, :, :, 0].detach().cpu().numpy(),
                    attn_weights=patch_attn[idx].detach().cpu().numpy(),
                    log_str="patch",
                    step_metric={"step": step},
                    mode=mode,
                    aggr_method="cls"
                    if config.spatial_aggr_method == "cls"
                    and config.temporal_aggr_method == "cls"
                    else "not cls lol",
                    test=test,
                    label=label,
                    frame_attn=frame_attn[0].squeeze().cpu().numpy(),
                    frame_num=idx,
                    sample_idx=sample_idx,
                )
        else:
            plot_attn_with_img(
                epoch=epoch,
                iteration=iteration,
                use_wandb=use_wandb,
                save_dir=save_dir,
                img=vids[0, 0, 0, :, :, 0].detach().cpu().numpy(),
                attn_weights=patch_attn.cpu().numpy(),
                log_str="patch",
                step_metric={"step": step},
                mode=mode,
                aggr_method=config["spatial_aggr_method"],
            )

    # Plot frame level attention (And indicate ED/ES locations if available)
    if test:
        frame_attn = frame_attn[0]
        vid_attn = vid_attn[0]

    plot_attn_map(
        epoch=epoch,
        iteration=iteration,
        use_wandb=use_wandb,
        save_dir=save_dir,
        attn_weights=frame_attn.cpu().numpy()[np.where(ed_valid == True)[0]][0]
        if (ed_valid is not None and np.any(ed_valid))
        else frame_attn.squeeze().cpu().numpy(),
        log_str="frame",
        step_metric={"step": step},
        mode=mode,
        aggr_method=config["temporal_aggr_method"],
        ed_frame=ed_frame[np.where(ed_valid == True)[0]]
        if (ed_valid is not None and np.any(ed_valid))
        else None,
        ed_valid=True if (ed_valid is not None and np.any(ed_valid)) else None,
        es_frame=es_frame[np.where(ed_valid == True)[0]]
        if (ed_valid is not None and np.any(ed_valid))
        else None,
        es_valid=es_valid[np.where(ed_valid == True)[0]]
        if (ed_valid is not None and np.any(ed_valid))
        else None,
    )

    # Plot the vid level attention
    plot_attn_map(
        epoch=epoch,
        iteration=iteration,
        use_wandb=use_wandb,
        save_dir=save_dir,
        attn_weights=vid_attn.cpu().numpy(),
        log_str="vid",
        step_metric={"step": step},
        mode=mode,
        aggr_method=config["vid_aggr_method"],
    )


def plot_attn_map(
    epoch,
    iteration,
    attn_weights,
    log_str,
    step_metric,
    mode,
    aggr_method,
    use_wandb,
    save_dir,
    ed_frame=None,
    ed_valid=None,
    es_frame=None,
    es_valid=None,
):
    step_name, step_value = step_metric.popitem()

    v, _ = attention_rollout(attn_weights)

    if aggr_method == "cls":
        fig, ax = plt.subplots(1, 1)
        ax.plot(v[0, 1:])
    else:
        fig, ax = plt.subplots(1)
        ax.imshow(v, cmap="hot")

    if ed_frame is not None:
        if ed_valid:
            plt.axvline(x=ed_frame.item(), color="r", label="axvline - full height")

        if es_valid:
            plt.axvline(x=es_frame.item(), color="b", label="axvline - full height")

    if use_wandb:
        wandb.log(
            {
                f"{mode}/" + log_str + "_attn": wandb.Image(fig),
                f"{mode}/{step_name}": step_value,
            }
        )
    else:
        plt.savefig(
            os.path.join(
                save_dir,
                "visualizations",
                "{}_level".format(log_str),
                "{}_{}_attn_{}_{}.png".format(mode, log_str, epoch, iteration),
            ),
        )

    plt.close("all")


def attention_rollout(attn_weights):
    # Add residual connections
    residual_attn = np.eye(attn_weights.shape[-1])
    aug_attn_weights = attn_weights + residual_attn

    # Normalize
    aug_attn_weights = aug_attn_weights / np.expand_dims(
        np.sum(aug_attn_weights, axis=-1), -1
    )

    joint_attn_weights = np.zeros_like(aug_attn_weights)
    joint_attn_weights[0] = aug_attn_weights[0]

    for n in range(1, aug_attn_weights.shape[0]):
        joint_attn_weights[n] = np.matmul(
            aug_attn_weights[n], joint_attn_weights[n - 1]
        )

    v = joint_attn_weights[-1]

    return v, aug_attn_weights


def transformer_vis(
    epoch,
    iteration,
    use_wandb,
    save_dir,
    vids,
    patch_attn,
    frame_attn,
    vid_attn,
    patch_pos,
    frame_pos,
    vid_pos,
    wandb_mode,
    config,
    num_steps,
    epoch_steps,
    batch_size,
    wandb_log_steps,
    ed_frame=None,
    ed_valid=None,
    es_frame=None,
    es_valid=None,
    num_frames=32,
    label=None,
    sample_idx=0,
    test=False,
):
    step = (epoch * epoch_steps + iteration) * batch_size

    if num_steps % wandb_log_steps == 0:
        plot_all_pos_embeds(
            epoch=epoch,
            iteration=iteration,
            use_wandb=use_wandb,
            save_dir=save_dir,
            patch_pos=patch_pos,
            frame_pos=frame_pos,
            vid_pos=vid_pos,
            config=config,
            step=step,
            mode=wandb_mode,
        )

        plot_all_attn_maps(
            epoch=epoch,
            iteration=iteration,
            use_wandb=use_wandb,
            save_dir=save_dir,
            vids=vids,
            patch_attn=patch_attn,
            frame_attn=frame_attn,
            vid_attn=vid_attn,
            step=step,
            mode=wandb_mode,
            config=config,
            ed_frame=ed_frame,
            ed_valid=ed_valid,
            es_frame=es_frame,
            es_valid=es_valid,
            test=test,
            num_frames=num_frames,
            label=label,
            sample_idx=sample_idx,
        )

    plt.close("all")
    return

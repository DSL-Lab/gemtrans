import torch
import random
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
from matplotlib import cm


def get_discard_img(view_img, discard_indices, fea_size, patch_size, replace_color):
    res_img = np.copy(view_img)
    for discard_indice in discard_indices:
        indice_h, indice_w = discard_indice // fea_size, discard_indice % fea_size
        res_img[indice_h * patch_size: (indice_h + 1) * patch_size,
        indice_w * patch_size: (indice_w + 1) * patch_size] = replace_color
    return res_img


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def get_gaussian_params(proto_act, scale_coe=0.9):
    fea_size = proto_act.shape[-1]
    discrete_values = np.array([[x, y] for x in range(fea_size) for y in range(fea_size)])
    discrete_values = discrete_values.transpose(1, 0)  # (2, 196)
    weights = proto_act.flatten()[np.newaxis, :]  # (1, 196)
    weights = weights / weights.sum(axis=-1)
    weights *= fea_size * fea_size
    # weights = (weights / weights.sum(axis=-1)) * scale_coe  # (1, 196)

    value_mean = np.mean(discrete_values * weights, axis=-1)  # (2,)
    cut_value = discrete_values - value_mean[:, np.newaxis]
    value_cov = np.dot(cut_value * weights, cut_value.transpose(1, 0))
    value_cov /= (fea_size * fea_size - 1)  # (2, 2)

    return value_mean, value_cov


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def save_fig(X, Y, Z, save_path):
    # scale Z
    Z = Z * 100

    # plot using subplots
    fea_size = X.shape[0]
    fig = plt.figure()
    ax1 = fig.gca(projection='3d')

    surf = ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=2, antialiased=True,
                            cmap=cm.viridis)
    # ax1.view_init(elev=75, azim=0)
    # ax1.view_init(elev=45, azim=20)
    ax1.view_init(elev=10, azim=20)

    ax1.set_xticks(np.arange(0, 14, 4))
    ax1.set_yticks(np.arange(0, 14, 4))
    # ax1.set_zticks(np.array([4, 8]))
    ax1.spines['bottom'].set_linewidth(8)
    ax1.spines['left'].set_linewidth(8)

    f_size = 20
    ax1.set_xlabel(r'$x^2$', fontsize=f_size, labelpad=12)
    ax1.set_ylabel(r'$x^1$', fontsize=f_size, labelpad=12)
    ax1.set_zlabel(r'similarity score', fontsize=20, labelpad=5)
    # fig.colorbar(surf, location='right', shrink=0.5, aspect=5)
    plt.subplots_adjust(left=0, bottom=0.05, right=1, top=0.95, hspace=0.1, wspace=0.1)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)
    ax1.tick_params('z', labelsize=20)

    # figure = plt.gcf()
    plt.savefig(save_path, pad_inches=-1)
    plt.clf()


def str2bool(v):
    if v.lower() in ("true", "yes", "t", "y"):
        return True
    elif v.lower() in ("false", "no", "f", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def update_prototypes_on_batch(search_batch_input, start_index_of_search_batch,
                               model,
                               global_max_proto_dist,  # this will be updated
                               global_max_fmap_patches,  # this will be updated
                               class_specific=True,
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               dir_for_saving_prototypes=None,
                               ):
    model.eval()
    search_batch = search_batch_input

    # select_colors = np.array([[47, 243, 224], [250, 38, 160], [248, 210, 16], [245, 23, 32]])
    token_reserve_num = 81

    with torch.no_grad():

        (vids, y, mask, lv_mask, ed_frame_idx, ed_valid, es_frame_idx, es_valid) = search_batch
        device = torch.device("cuda")
        vids = vids.to(device)
        mask = mask.to(device)
        lv_mask = lv_mask.to(device)
        ed_frame_idx = ed_frame_idx.to(device)
        ed_valid = ed_valid.to(device)
        es_frame_idx = es_frame_idx.to(device)
        es_valid = es_valid.to(device)
        y = y.to(device)
        (
            _, _, _, _, _, _, _, _, _, _, _, _, _, _,
            logits,
            token_attn,
            distances,
            _
        ) = model.module(vids, mask, ed_frame_idx, ed_valid, es_frame_idx, es_valid)
        # token_attn, distances = auxi_items[0], auxi_items[1] # (B,196),(B, 32, 11, 11)
        _, pred = logits.topk(k=1, dim=1)
        token_attn, distances = token_attn.unsqueeze(0), distances.unsqueeze(0)

    patch_size = 16
    proto_per_category = 8

    # Save video in uint format
    video = (((0.5 * vids[0, 0]) + 0.5) * 255).expand(-1, -1, -1, 3)
    video = video.detach().cpu().numpy().astype(np.uint8)

    # this computation currently is not parallelized
    all_token_attn = np.copy(token_attn.detach().cpu().numpy())
    min_distances = np.copy(distances.detach().cpu().numpy())
    del token_attn, distances

    label = y.cpu().numpy()
    pred = pred.cpu().numpy()
    is_pred_right = (label == pred)

    proto_acts = np.log((min_distances + 1) / (min_distances + model.module.prototype_layer.epsilon))
    total_proto_acts = proto_acts
    proto_acts = np.amax(proto_acts, (3, 4))

    sample_num, num_frames, num_prototypes = proto_acts.shape[0], proto_acts.shape[1], proto_acts.shape[2]

    fea_size = int(all_token_attn.shape[-1] ** (1 / 2))
    token_attn = all_token_attn.reshape(all_token_attn.shape[0], all_token_attn.shape[1], -1)
    token_attn = torch.from_numpy(token_attn)

    # 9 * 9 -> 14 * 14 fill into
    total_proto_acts = torch.from_numpy(total_proto_acts)
    reserve_token_indices = torch.topk(token_attn, k=token_reserve_num, dim=-1)[1]
    reserve_token_indices = reserve_token_indices.sort(dim=-1)[0]
    reserve_token_indices = reserve_token_indices[:, :, None, :].repeat(1, 1, num_prototypes, 1)  # (B, 2000, 81)
    replace_proto_acts = torch.zeros(sample_num, num_frames, num_prototypes, 196)
    replace_proto_acts.scatter_(3, reserve_token_indices, total_proto_acts.flatten(start_dim=3))  # (B, 2000, 196)
    replace_proto_acts = replace_proto_acts.reshape(sample_num, num_frames, num_prototypes, 14, 14).numpy()

    visual_type = 'slim_gaussian'
    proto_per_category = int(num_prototypes / num_classes)
    for proto_idx in range(proto_per_category):
        # for proto_idx in range(1):
        # select view_imgs, proto_acts, labels
        # print('process proto {}...'.format(proto_idx))
        cur_proto_acts = replace_proto_acts[:, :, label[0] * proto_per_category + proto_idx]
        cur_max_proto_dist = np.amax(cur_proto_acts)
        if cur_max_proto_dist > global_max_proto_dist[label[0] * proto_per_category + proto_idx]:
            global_max_proto_dist[label[0] * proto_per_category + proto_idx] = cur_max_proto_dist
            cur_argmax_proto_dist = np.unravel_index(np.argmax(cur_proto_acts, axis=None), cur_proto_acts.shape)
            f = cur_argmax_proto_dist[1]

            proto_act = cur_proto_acts[0]
            upsampled_act = cv2.resize(proto_act[f], (224, 224), interpolation=cv2.INTER_CUBIC)
            upsampled_act = upsampled_act - np.amin(upsampled_act)
            upsampled_act = upsampled_act / np.amax(upsampled_act)

            # get the top 5% bounding box
            coor = find_high_activation_crop(upsampled_act)

            heatmap = cv2.applyColorMap(np.uint8(255 * upsampled_act), cv2.COLORMAP_JET)
            acti_vid = (video[f] * 0.7 + heatmap * 0.3).astype(np.uint8)

            bnd_img = video[f]
            start_point, end_point = (coor[2], coor[0]), (coor[3], coor[1])  # (x coor, y coor)
            bnd_img = cv2.rectangle(bnd_img.astype(np.uint8).copy(), start_point, end_point, (0, 255, 255), thickness=2)

            num_patches = token_attn.shape[-1]
            replace_color = [0, 0, 0]
            discard_token_indices = torch.topk(token_attn, k=num_patches - token_reserve_num, dim=-1, largest=False)[1]
            cur_discard_indices = discard_token_indices[0][f].numpy()
            discard_img = get_discard_img(video[f], cur_discard_indices, fea_size, patch_size, replace_color)

            #### PLOTTING
            fig, axs = plt.subplots(1, 4, figsize=(20, 6))
            axs[0].imshow(video[f])  # TODO denomarlize maybe using our normalization?
            axs[0].title.set_text('original')
            axs[1].imshow(discard_img)  # TODO denomarlize maybe using our normalization?
            axs[1].title.set_text('Discarded')
            axs[2].imshow(bnd_img)  # TODO deomarlize maybe using our normalization?
            axs[2].title.set_text('ROI')
            axs[3].imshow(acti_vid)  # TODO deomarlize maybe using our normalization?
            axs[3].title.set_text('heatmap')
            fig.tight_layout()
            plt.axis('off')
            plt.savefig(os.path.join(dir_for_saving_prototypes, 'class{}_proto{}'.format(label[0], label[
                0] * proto_per_category + proto_idx) + '.png'))
            plt.close()

    return global_max_proto_dist

    # # Generating activation map with JET colormap F*14*14*3
    # new_proto_act = cur_proto_acts[0]
    # proto_act_vid = None
    # for f in range(num_frames):
    #     new_proto_act[f] = new_proto_act[f] - np.expand_dims(np.amin(new_proto_act[f],axis=(0,1)),axis=(0,1))
    #     new_proto_act[f] = new_proto_act[f] / np.expand_dims(np.amax(new_proto_act[f],axis=(0,1)),axis=(0,1))
    #     new_proto_act_v = cv2.applyColorMap(np.uint8(255 * new_proto_act[f]), cv2.COLORMAP_JET)
    #     if proto_act_vid is None:
    #         proto_act_vid = np.expand_dims(new_proto_act_v,axis=0)
    #     else:
    #         proto_act_vid = np.concatenate([proto_act_vid ,np.expand_dims(new_proto_act_v,axis=0)], axis=0)

    # # Heatmap in 224*224
    # heatmap_vid, patch_idx_vid, coor_vid= None, None, None
    # proto_act = cur_proto_acts[0]
    # for f in range(num_frames):
    #     upsampled_act = cv2.resize(proto_act[f], (224, 224), interpolation=cv2.INTER_CUBIC)
    #     upsampled_act = upsampled_act - np.amin(upsampled_act)
    #     upsampled_act = upsampled_act / np.amax(upsampled_act)

    #     # get the top 5% bounding box
    #     coor = find_high_activation_crop(upsampled_act)

    #     heatmap = cv2.applyColorMap(np.uint8(255 * upsampled_act), cv2.COLORMAP_JET)
    #     patch_idx = [t[0] for t in np.where(proto_act[f] == proto_act[f].max())]
    #     if heatmap_vid is None:
    #         coor_vid = np.expand_dims(coor,axis=0)
    #         heatmap_vid = np.expand_dims(heatmap,axis=0)
    #         patch_idx_vid = np.expand_dims(patch_idx,axis=0)
    #     else:
    #         coor_vid = np.concatenate([coor_vid ,np.expand_dims(coor,axis=0)], axis=0)
    #         heatmap_vid = np.concatenate([heatmap_vid ,np.expand_dims(heatmap,axis=0)], axis=0)
    #         patch_idx_vid = np.concatenate([patch_idx_vid ,np.expand_dims(patch_idx,axis=0)], axis=0)

    # acti_vid = (video * 0.7 + heatmap_vid * 0.3).astype(np.uint8)

    # # view the masks
    # if visual_type == 'slim_gaussian':
    #     bnd_vid = []
    #     # draw the bounding boxes
    #     for f in range(num_frames):
    #         bnd_img = video[f]
    #         coor_f = coor_vid[f]
    #         start_point, end_point = (coor_f[2], coor_f[0]), (coor_f[3], coor_f[1]) # (x coor, y coor)

    #         # part_img = bnd_img[coor[0]:coor[1], coor[2]:coor[3]]
    #         # cv2.imwrite(os.path.join(img_dir, 'proto{}_reserve{}_part.jpg'.format(proto_idx, token_reserve_num)), part_img)
    #         bnd_img = cv2.rectangle(bnd_img.astype(np.uint8).copy(), start_point, end_point, (0, 255, 255), thickness=2)
    #         bnd_vid.append(bnd_img)

    #     num_patches = token_attn.shape[-1]
    #     replace_color = [0, 0, 0]
    #     discard_token_indices = torch.topk(token_attn, k=num_patches - token_reserve_num, dim=-1, largest=False)[1]
    #     discard_vid = None
    #     for f in range(num_frames):
    #         cur_discard_indices = discard_token_indices[0][f].numpy()
    #         discard_img = get_discard_img(video[f], cur_discard_indices, fea_size, patch_size, replace_color)
    #         if discard_vid is None:
    #             discard_vid = np.expand_dims(discard_img,axis=0)
    #         else:
    #             discard_vid = np.concatenate([discard_vid,np.expand_dims(discard_img,axis=0)], axis=0)

    # from matplotlib import rc
    # rc('animation', html='jshtml')
    # fig, ax = plt.subplots()
    # frames = [[ax.imshow(video[s])] for s in range(1,32)]
    # path_to_save = os.path.join(dir_for_saving_prototypes,
    #                     'class{}_proto{}_video'.format(label[0],label[0] * proto_per_category + proto_idx))
    # ani = animation.ArtistAnimation(fig, frames)
    # writergif = animation.PillowWriter(fps=15)
    # ani.save(path_to_save,writer=writergif)

    # fig, ax = plt.subplots()
    # frames = [[ax.imshow(discard_vid[s])] for s in range(1,32)]
    # path_to_save = os.path.join(dir_for_saving_prototypes,
    #                     'class{}_proto{}_discard' + str(label[0],label[0] * proto_per_category + proto_idx))
    # ani = animation.ArtistAnimation(fig, frames)
    # writergif = animation.PillowWriter(fps=15)
    # ani.save(path_to_save,writer=writergif)

    # fig, ax = plt.subplots()
    # frames = [[ax.imshow(bnd_vid[s])] for s in range(1,32)]
    # path_to_save = os.path.join(dir_for_saving_prototypes,
    #                     'class{}_proto{}_rectangle' + str(label[0],label[0] * proto_per_category + proto_idx))
    # ani = animation.ArtistAnimation(fig, frames)
    # writergif = animation.PillowWriter(fps=15)
    # ani.save(path_to_save,writer=writergif)





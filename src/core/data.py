import torchvision.transforms
import numpy as np
import SimpleITK as sitk
from torch.nn.utils.rnn import pad_sequence
import re
from operator import itemgetter
from math import isnan
import cv2
import random
import math
import collections
import skimage.draw
import torch.nn.functional as F
import bz2
import _pickle as cPickle
import scipy.io as spio
import os
from os.path import join
from typing import Callable, Dict
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import json
from glob import glob
from torch.nn import Upsample
import itertools
from torchvision.datasets import Kinetics

random.seed(0)
np.random.seed(0)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class EchoNetAp4Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        mode,
        max_frames=32,
        transform=None,
        split="train",
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        classification_classes = np.array([0, 30, 40, 55, 100])

        assert split in ["train", "val", "test"]

        # CSV file containing file names and labels
        file_list_df = pd.read_csv(os.path.join(dataset_path, "FileList.csv"))

        # Extract Split information
        splits = np.array(file_list_df["Split"].tolist())
        self.train_idx = np.where(splits == "TRAIN")[0]
        self.val_idx = np.where(splits == "VAL")[0]
        self.test_idx = np.where(splits == "TEST")[0]

        # Keep the correct slit
        if split == "train":
            file_list_df = file_list_df.loc[file_list_df["Split"] == "TRAIN"]
        elif split == "val":
            file_list_df = file_list_df.loc[file_list_df["Split"] == "VAL"]
        elif split == "test":
            file_list_df = file_list_df.loc[file_list_df["Split"] == "TEST"]

        # Get the list of video names and their ef values
        file_names = file_list_df["FileName"].tolist()
        labels = file_list_df["EF"].tolist()

        # Extract ES and ED frame indices
        es_frames = file_list_df["ESFrame"].tolist()
        self.es_frames = [
            None if isnan(es_frame) else int(es_frame) for es_frame in es_frames
        ]
        ed_frames = file_list_df["EDFrame"].tolist()
        self.ed_frames = [
            None if isnan(ed_frame) else int(ed_frame) for ed_frame in ed_frames
        ]

        # Extract LV segmentation masks
        if use_seg_labels:
            file_names, labels = self._extract_lv_trace(
                dataset_path, file_names, labels
            )

        # Full file paths
        self.patient_data_dirs = [
            os.path.join(dataset_path, "Videos", file_name + ".avi")
            for file_name in file_names
        ]

        # Categorize EF values
        self.classification_labels = (
            np.digitize(np.array(labels), classification_classes) - 1
        )
        self.classification_labels = torch.tensor(
            self.classification_labels, dtype=torch.long
        )

        # Bring EF values to [0,1]
        self.labels = list()
        for patient, _ in enumerate(self.patient_data_dirs):
            self.labels.append(labels[patient] / 100)

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Other attribues
        self.trans = transform
        self.aug_trans = aug_transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.max_frames = max_frames
        self.mode = mode
        self.train = split == "train"
        self.use_seg_labels = use_seg_labels
        self.max_clips = max_clips
        self.mean_std = mean_std

    def __getitem__(self, idx):

        # If the dataset is only created to find its mean and std
        if self.mean_std:
            ap4_cine_vid = self._loadvideo(self.patient_data_dirs[idx])
            ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))
            return ap4_cine_vid

        # extract cine vids
        ap4_cine_vid = self._loadvideo(self.patient_data_dirs[idx])
        orig_size = ap4_cine_vid.shape[0]

        # Mask indicating which frames are padding
        mask = torch.ones((1, self.max_frames), dtype=torch.bool)

        # Pad the video and extract clips
        ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))
        (
            ap4_cine_vid,
            mask,
            lv_mask,
            ed_frame,
            ed_valid,
            es_frame,
            es_valid,
        ) = self._pad_vid(ap4_cine_vid, mask, idx, orig_size)

        # Perform augmentation during training
        if self.train and self.aug_trans is not None:
            ap4_cine_vid = self.aug_trans(ap4_cine_vid)

        # Interpolate the image
        if self.use_seg_labels:
            lv_mask = F.interpolate(
                lv_mask.unsqueeze(0).unsqueeze(1),
                size=(ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-1]),
            )
            lv_mask = lv_mask.squeeze(0)

        return {
            "vid": ap4_cine_vid.unsqueeze(0).unsqueeze(-1)
            if self.train
            else ap4_cine_vid.unsqueeze(1).unsqueeze(-1),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "mask": mask,
            "lv_mask": lv_mask,
            "ed_frame": torch.tensor(ed_frame),
            "ed_valid": torch.tensor(ed_valid),
            "es_frame": torch.tensor(es_frame),
            "es_valid": torch.tensor(es_valid),
            "class_label": self.classification_labels[idx],
        }

    def _extract_lv_trace(self, dataset_path, file_names, labels):
        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        with open(os.path.join(dataset_path, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                frame = int(frame)
                if frame not in self.trace[filename]:
                    self.frames[filename].append(frame)
                self.trace[filename][frame].append((x1, y1, x2, y2))
        for filename in self.frames:
            for frame in self.frames[filename]:
                self.trace[filename][frame] = np.array(self.trace[filename][frame])

        keep = [len(self.frames[f + ".avi"]) >= 2 for f in file_names]
        file_names = [f for (f, k) in zip(file_names, keep) if k]
        labels = [f for (f, k) in zip(labels, keep) if k]
        self.ed_frames = [f for (f, k) in zip(self.ed_frames, keep) if k]
        self.es_frames = [f for (f, k) in zip(self.es_frames, keep) if k]

        return file_names, labels

    def _pad_vid(self, vid, mask, patient_idx, orig_size=None):

        file_name = os.path.basename(self.patient_data_dirs[patient_idx])

        # Combine the LV mask for ED and ES frames
        lv_mask_collated = torch.zeros(1)
        if self.use_seg_labels:
            for i in range(2):
                t = self.trace[file_name][self.frames[file_name][i]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(
                    np.rint(y).astype(np.int),
                    np.rint(x).astype(np.int),
                    (orig_size, orig_size),
                )
                lv_mask = np.zeros((orig_size, orig_size), np.bool)
                lv_mask[r, c] = 1
                lv_mask_collated = (
                    lv_mask if i == 0 else np.bitwise_or(lv_mask_collated, lv_mask)
                )
            lv_mask_collated = torch.from_numpy(lv_mask_collated.astype(np.float32))

        # If the number of frames is less than max frames, pad with 0's
        if vid.shape[0] <= self.max_frames:
            mask[0, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )

            ed_frame_idx, ed_valid, es_frame_idx, es_valid = self._frame_idx_in_clip(
                patient_idx, np.arange(self.max_frames)
            )

            if not self.train:
                mask = mask.unsqueeze(0)
                vid = vid.unsqueeze(0)
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)

                (
                    ed_frame_idx,
                    ed_valid,
                    es_frame_idx,
                    es_valid,
                ) = self._frame_idx_in_clip(
                    patient_idx, np.arange(starting_idx, starting_idx + self.max_frames)
                )

                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                # During validation and testing use all available clips
                ed_valid = []
                ed_frame_idx = []
                es_valid = []
                es_frame_idx = []

                num_clips = min(
                    math.ceil(vid.shape[0] / self.max_frames), self.max_clips
                )

                curated_clips = None
                for clip_idx in range(num_clips - 1):
                    curated_clips = (
                        vid[0 : self.max_frames].unsqueeze(0)
                        if curated_clips is None
                        else torch.cat(
                            (
                                curated_clips,
                                vid[
                                    self.max_frames
                                    * clip_idx : self.max_frames
                                    * (clip_idx + 1)
                                ].unsqueeze(0),
                            ),
                            dim=0,
                        )
                    )

                    (
                        clip_ed_idx,
                        clip_ed_valid,
                        clip_es_idx,
                        clip_es_valid,
                    ) = self._frame_idx_in_clip(
                        patient_idx,
                        np.arange(
                            self.max_frames * clip_idx, self.max_frames * (clip_idx + 1)
                        ),
                    )

                    ed_valid.append(clip_ed_valid)
                    ed_frame_idx.append(clip_ed_idx)
                    es_valid.append(clip_es_valid)
                    es_frame_idx.append(clip_es_idx)

                # The last clip is allowed to overlap with the previous one
                curated_clips = (
                    vid[-self.max_frames :].unsqueeze(0)
                    if curated_clips is None
                    else torch.cat(
                        (curated_clips, vid[-self.max_frames :].unsqueeze(0)), dim=0
                    )
                )

                (
                    clip_ed_idx,
                    clip_ed_valid,
                    clip_es_idx,
                    clip_es_valid,
                ) = self._frame_idx_in_clip(
                    patient_idx,
                    np.arange(vid.shape[0] - self.max_frames, vid.shape[0]),
                )

                ed_valid.append(clip_ed_valid)
                ed_frame_idx.append(clip_ed_idx)
                es_valid.append(clip_es_valid)
                es_frame_idx.append(clip_es_idx)

                vid = curated_clips
                mask = torch.cat([mask.unsqueeze(0)] * num_clips, dim=0)

        return (
            vid,
            mask,
            lv_mask_collated,
            ed_frame_idx,
            ed_valid,
            es_frame_idx,
            es_valid,
        )

    def _frame_idx_in_clip(self, data_idx, clip_idx):
        ed_frame, ed_valid, es_frame, es_valid = 0, False, 0, False

        if self.ed_frames[data_idx] in clip_idx:
            ed_frame = np.where(clip_idx == self.ed_frames[data_idx])[0].item()
            ed_valid = True

        if self.es_frames[data_idx] in clip_idx:
            es_frame = np.where(clip_idx == self.es_frames[data_idx])[0].item()
            es_valid = True

        return ed_frame, ed_valid, es_frame, es_valid

    def __len__(self):
        """
        Returns number of available samples

        :return: Number of graphs
        """
        return self.num_samples

    @staticmethod
    def _loadvideo(filename: str):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError(
                    "Failed to load frame #{} of {}.".format(count, filename)
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v


class LVBiplaneEFDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        mode,
        max_frames=32,
        transform=None,
        split="train",
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        assert mode == "ef", "Currently only EF is supported for this dataset."

        classification_classes = np.array([0, 0.30, 0.40, 0.55, 1.0])

        # CSV file containing file names and labels
        report_df = pd.read_csv(os.path.join(dataset_path, "report.csv"))
        dicom_df = pd.read_csv(os.path.join(dataset_path, "dicom_info.csv"))

        # # Keep the correct slit
        if split == "train":
            report_df = report_df.loc[report_df["Split"] == "Train"]
        elif split == "val":
            report_df = report_df.loc[report_df["Split"] == "Val"]
        elif split == "test":
            report_df = report_df.loc[report_df["Split"] == "Test"]

        # Get patient IDs
        patient_ids = np.array(report_df["PatientID"].tolist())

        # Labels list
        self.labels = list()

        # Find path to available videos for each patient
        self.patient_data_dirs = dict()
        patient_num = 0
        for patient_id in patient_ids:
            file_paths = dicom_df.loc[
                dicom_df["PatientID_o"] == patient_id, "cleaned_filepath"
            ].tolist()

            # Have to change from pbz2 to mat since the dataset structure has changed since when the report csv was
            # created
            file_paths = [
                os.path.join(
                    dataset_path,
                    "batch_1",
                    *os.path.normpath(file_path).split("\\")[-3:],
                )[0:-4]
                + "mat"
                for file_path in file_paths
            ]

            all_files_exist = len(file_paths) > 1
            for file in file_paths:
                if not os.path.exists(file):
                    all_files_exist = False

            label_exists = (
                len(report_df.loc[report_df["PatientID"] == patient_id, "EF"].tolist())
                > 0
            )

            if all_files_exist and label_exists:
                self.patient_data_dirs[patient_num] = file_paths
                self.labels.append(
                    report_df.loc[report_df["PatientID"] == patient_id, "EF"].tolist()[
                        0
                    ]
                    / 100
                )
                patient_num += 1
            else:
                report_df = report_df.drop(
                    report_df[report_df["PatientID"] == patient_id].index
                )

        self.classification_labels = (
            np.digitize(np.array(self.labels), classification_classes) - 1
        )
        self.classification_labels = torch.tensor(
            self.classification_labels, dtype=torch.long
        )

        # Extract Split information
        splits = np.array(report_df["Split"].tolist())
        self.train_idx = np.where(splits == "Train")[0]
        self.val_idx = np.where(splits == "Val")[0]
        self.test_idx = np.where(splits == "Test")[0]

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Other attributes
        self.max_frames = max_frames
        self.mode = mode
        self.trans = transform
        self.train = split == "train"
        self.use_seg_labels = use_seg_labels
        self.max_clips = max_clips
        self.aug_trans = aug_transform
        self.mean_std = mean_std

    def __getitem__(self, idx):

        # Extract the AP4 Video and its labels
        data = self.loadmat(self.patient_data_dirs[idx][0])
        ap4_cine_vid = data["cropped"]

        if self.mean_std:
            return self.trans(np.array(ap4_cine_vid, dtype=np.uint8))

        ap4_ed_seg = None
        ap4_es_seg = None
        ap2_ed_seg = None
        ap2_es_seg = None

        ed_valid = [False, False]
        ed_frame_idx = [0, 0]
        es_valid = [False, False]
        es_frame_idx = [0, 0]
        try:
            for key in data["labels"]:
                if "LV_vol_d" in key:
                    ed_frame_idx[1] = data["labels"][key]["frame_num"] - 1
                    ed_valid[1] = True
                    ap4_ed_seg = data["labels"][key]["trace"]["mask_cropped"]
                elif "LV_vol_s" in key:
                    es_frame_idx[1] = data["labels"][key]["frame_num"] - 1
                    es_valid[1] = True
                    ap4_es_seg = data["labels"][key]["trace"]["mask_cropped"]
        except KeyError:
            print(
                "{} does not contain frame location labels. "
                "Setting all frame locations to 0.".format(
                    self.patient_data_dirs[idx][0]
                )
            )

        if ap4_ed_seg is None or ap4_es_seg is None:
            ap4_mask = None
        else:
            ap4_mask = torch.from_numpy(
                np.bitwise_or(ap4_ed_seg, ap4_es_seg).astype(np.float32)
            )

        # Extract the AP2 Video and its labels
        data = self.loadmat(self.patient_data_dirs[idx][1])
        ap2_cine_vid = data["cropped"]

        try:
            for key in data["labels"]:
                if "LV_vol_d" in key:
                    ed_frame_idx[0] = data["labels"][key]["frame_num"] - 1
                    ed_valid[0] = True
                    ap2_ed_seg = data["labels"][key]["trace"]["mask_cropped"]
                elif "LV_vol_s" in key:
                    es_frame_idx[0] = data["labels"][key]["frame_num"] - 1
                    es_valid[0] = True
                    ap2_es_seg = data["labels"][key]["trace"]["mask_cropped"]
        except KeyError:
            print(
                "{} does not contain frame location labels. "
                "Setting all frame locations to 0.".format(
                    self.patient_data_dirs[idx][1]
                )
            )

        if ap2_ed_seg is None or ap2_es_seg is None:
            ap2_mask = None
        else:
            ap2_mask = torch.from_numpy(
                np.bitwise_or(ap2_ed_seg, ap2_es_seg).astype(np.float32)
            )

        # During validation/test time, extract multiple clips per video
        num_clips = min(
            math.ceil(
                max(ap2_cine_vid.shape[-1], ap4_cine_vid.shape[-1]) / self.max_frames
            ),
            self.max_clips,
        )
        if not self.train:
            mask = torch.ones((num_clips, 2, self.max_frames), dtype=torch.bool)
            ap2_temp = list()
            ap4_temp = list()

            ed_valid = np.stack([ed_valid for _ in range(num_clips)])
            ed_frame_idx = np.stack([ed_frame_idx for _ in range(num_clips)])
            es_valid = np.stack([es_valid for _ in range(num_clips)])
            es_frame_idx = np.stack([es_frame_idx for _ in range(num_clips)])

            ap2_cine_vid = self.trans(np.array(ap2_cine_vid, dtype=np.uint8))
            ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))

            for clip_idx in range(num_clips):
                (
                    ap2_cine_vid_temp,
                    mask[clip_idx],
                    ed_frame_idx[clip_idx][0],
                    ed_valid[clip_idx][0],
                    es_frame_idx[clip_idx][0],
                    es_valid[clip_idx][0],
                ) = self.pad_vid(
                    ap2_cine_vid,
                    mask[clip_idx],
                    0,
                    ed_frame_idx[clip_idx][0],
                    ed_valid[clip_idx][0],
                    es_frame_idx[clip_idx][0],
                    es_valid[clip_idx][0],
                    clip_idx,
                )
                ap2_temp.append(ap2_cine_vid_temp.unsqueeze(0))

                (
                    ap4_cine_vid_temp,
                    mask[clip_idx],
                    ed_frame_idx[clip_idx][1],
                    ed_valid[clip_idx][1],
                    es_frame_idx[clip_idx][1],
                    es_valid[clip_idx][1],
                ) = self.pad_vid(
                    ap4_cine_vid,
                    mask[clip_idx],
                    1,
                    ed_frame_idx[clip_idx][1],
                    ed_valid[clip_idx][1],
                    es_frame_idx[clip_idx][1],
                    es_valid[clip_idx][1],
                    clip_idx,
                )
                ap4_temp.append(ap4_cine_vid_temp.unsqueeze(0))

            ap2_cine_vid = torch.cat(ap2_temp, dim=0)
            ap4_cine_vid = torch.cat(ap4_temp, dim=0)
            output = torch.cat(
                (
                    ap2_cine_vid.unsqueeze(1).unsqueeze(-1),
                    ap4_cine_vid.unsqueeze(1).unsqueeze(-1),
                ),
                dim=1,
            )
        else:
            mask = torch.ones((2, self.max_frames), dtype=torch.bool)
            ap2_cine_vid = self.trans(np.array(ap2_cine_vid, dtype=np.uint8))
            (
                ap2_cine_vid,
                mask,
                ed_frame_idx[0],
                ed_valid[0],
                es_frame_idx[0],
                es_valid[0],
            ) = self.pad_vid(
                ap2_cine_vid,
                mask,
                0,
                ed_frame_idx[0],
                ed_valid[0],
                es_frame_idx[0],
                es_valid[0],
            )

            ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))
            (
                ap4_cine_vid,
                mask,
                ed_frame_idx[1],
                ed_valid[1],
                es_frame_idx[1],
                es_valid[1],
            ) = self.pad_vid(
                ap4_cine_vid,
                mask,
                1,
                ed_frame_idx[1],
                ed_valid[1],
                es_frame_idx[1],
                es_valid[1],
            )

            output = torch.cat(
                (
                    ap2_cine_vid.unsqueeze(0).unsqueeze(-1),
                    ap4_cine_vid.unsqueeze(0).unsqueeze(-1),
                ),
                dim=0,
            )

            output = self.aug_trans(output)

        # Prepare the LV mask if needed
        if self.use_seg_labels:
            if not (ap2_mask is None or ap4_mask is None):
                ap2_lv_mask = F.interpolate(
                    ap2_mask.unsqueeze(0).unsqueeze(1),
                    size=(ap2_cine_vid.shape[-1], ap2_cine_vid.shape[-2]),
                ).squeeze(0)
                ap4_lv_mask = F.interpolate(
                    ap4_mask.unsqueeze(0).unsqueeze(1),
                    size=(ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-2]),
                ).squeeze(0)
                lv_mask = torch.cat((ap2_lv_mask, ap4_lv_mask), dim=0)

                if not self.train:
                    lv_mask = torch.cat(
                        [lv_mask.unsqueeze(0) for _ in range(num_clips)], dim=0
                    )
            else:
                if self.train:
                    lv_mask = torch.ones(
                        (2, ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-2])
                    )
                else:
                    lv_mask = torch.ones(
                        (num_clips, 2, ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-2])
                    )
        else:
            lv_mask = torch.zeros(1)

        ed_frame_idx = torch.tensor(ed_frame_idx, dtype=torch.long)
        ed_valid = torch.tensor(ed_valid, dtype=torch.bool)
        es_frame_idx = torch.tensor(es_frame_idx, dtype=torch.long)
        es_valid = torch.tensor(es_valid, dtype=torch.bool)

        return {
            "vid": output,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "mask": mask,
            "lv_mask": lv_mask,
            "ed_frame": ed_frame_idx,
            "ed_valid": ed_valid,
            "es_frame": es_frame_idx,
            "es_valid": es_valid,
            "class_label": self.classification_labels[idx],
        }

    def __len__(self):
        """
        Returns number of available samples

        :return: Number of graphs
        """

        return self.num_samples

    @staticmethod
    def decompress_pickle(file):
        """
        Decomporesses PBZ2 files
        Code from https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e

        :param file: str, path to file
        :return: decomporessed video
        """
        data = bz2.BZ2File(file, "rb")
        data = cPickle.load(data)
        return data

    # The code below is directly copied from
    # https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    def loadmat(self, filename):
        """
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        """
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict

    def pad_vid(
        self, vid, mask, mask_idx, ed_frame, ed_valid, es_frame, es_valid, clip_idx=0
    ):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )

            ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                np.arange(self.max_frames), ed_frame, ed_valid, es_frame, es_valid
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]

                ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                    np.arange(starting_idx, starting_idx + self.max_frames),
                    ed_frame,
                    ed_valid,
                    es_frame,
                    es_valid,
                )
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]

                    ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                        np.arange(
                            self.max_frames * clip_idx, self.max_frames * (clip_idx + 1)
                        ),
                        ed_frame,
                        ed_valid,
                        es_frame,
                        es_valid,
                    )
                else:
                    vid = vid[-self.max_frames :]

                    ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                        np.arange(vid.shape[0] - self.max_frames, vid.shape[0]),
                        ed_frame,
                        ed_valid,
                        es_frame,
                        es_valid,
                    )

        return vid, mask, ed_frame, ed_valid, es_frame, es_valid

    def _frame_idx_in_clip(self, clip_idx, ed_frame, ed_valid, es_frame, es_valid):

        if ed_frame in clip_idx and ed_valid:
            ed_frame = np.where(clip_idx == ed_frame)[0].item()
            ed_valid = True
        else:
            ed_valid = False

        if es_frame in clip_idx and es_valid:
            es_frame = np.where(clip_idx == es_frame)[0].item()
            es_valid = True
        else:
            es_valid = False

        return ed_frame, ed_valid, es_frame, es_valid


label_schemes = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}


def bicuspid_filter(df: pd.DataFrame):
    return df[~df["Bicuspid"]]


filtering_functions: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "bicuspid": bicuspid_filter
}


class AorticStenosisDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = "~/as",
        split: str = "train",
        mode: str = "as",
        max_frames: int = 16,
        transform=None,
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        assert mode == "as", "Only AS mode is supported"

        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])

        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(join(dataset_path, "annotations-all.csv"))
        # append dataset root to each path in the dataframe
        dataset["path"] = dataset["path"].map(lambda x: join(dataset_path, x))

        dataset = dataset[dataset["as_label"].map(lambda x: x in label_schemes.keys())]

        # Take train/test/val
        dataset = dataset[dataset.split == split]
        # Apply an arbitrary filter
        # filtering_function = filtering_functions["bicuspid"]
        # dataset = filtering_function(dataset)

        self.patient_studies = list()
        for patient_id in list(set(dataset["patient_id"])):
            for study_date in list(
                set(dataset[dataset["patient_id"] == patient_id]["date"])
            ):
                self.patient_studies.append((patient_id, study_date))

        self.dataset = dataset
        self.max_frames = max_frames
        self.train = split == "train"
        self.trans = transform
        self.aug_trans = aug_transform
        self.max_clips = max_clips
        self.mean_std = mean_std

    def class_samplers(self):
        labels_AS = list()
        for pid in self.patient_studies:
            patient_id, study_date = pid
            data_info = self.dataset[self.dataset["patient_id"] == patient_id]
            data_info = data_info[data_info["date"] == study_date]

            labels_AS.append(label_schemes[data_info["as_label"].iloc[0]])

        class_sample_count_AS = np.array(
            [len(np.where(labels_AS == t)[0]) for t in np.unique(labels_AS)]
        )
        weight_AS = 1.0 / class_sample_count_AS

        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS, 0, 0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()

        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))

        return sampler_AS

    def __len__(self) -> int:
        return len(self.patient_studies)

    def __getitem__(self, item):
        patient_id, study_date = self.patient_studies[item]
        data_info = self.dataset[self.dataset["patient_id"] == patient_id]
        data_info = data_info[data_info["date"] == study_date]
        # label = torch.tensor(self.labelling_scheme[data_info[self.label_key]])

        available_views = list(data_info["view"])
        num_plax = available_views.count("plax")
        num_psax = available_views.count("psax")

        frame_nums = list()

        if self.mean_std:
            return self.trans(
                np.moveaxis(loadmat(data_info["path"].iloc[0])["cine"], 0, -1)
            )

        all_plax_cine = list()
        if num_plax > 0:
            plax_indices = (
                [np.random.randint(num_plax)] if self.train else list(range(num_plax))
            )

            for plax_idx in plax_indices:
                plax_data_info = data_info[data_info["view"] == "plax"].iloc[plax_idx]

                # Transform and augment PLAX vid
                plax_cine = self.trans(
                    np.moveaxis(loadmat(plax_data_info["path"])["cine"], 0, -1)
                )
                if self.aug_trans is not None:
                    plax_cine = plax_cine.unsqueeze(0)
                    plax_cine = self.aug_trans(plax_cine)
                    plax_cine = plax_cine.squeeze(0)

                all_plax_cine.append(plax_cine)
                frame_nums.append(plax_cine.shape[0])

        all_psax_cine = list()
        if num_psax > 0:
            psax_indices = (
                [np.random.randint(num_psax)] if self.train else list(range(num_psax))
            )

            for psax_idx in psax_indices:
                psax_data_info = data_info[data_info["view"] == "psax"].iloc[psax_idx]

                # Transform and augment psax vid
                psax_cine = self.trans(
                    np.moveaxis(loadmat(psax_data_info["path"])["cine"], 0, -1)
                )
                if self.aug_trans is not None:
                    psax_cine = psax_cine.unsqueeze(0)
                    psax_cine = self.aug_trans(psax_cine)
                    psax_cine = psax_cine.squeeze(0)

                all_psax_cine.append(psax_cine)
                frame_nums.append(psax_cine.shape[0])

        no_plax = False
        no_psax = False
        if num_plax == 0:
            all_plax_cine.append(torch.zeros_like(all_psax_cine[0]))
            num_plax = 1
            no_plax = True
        elif num_psax == 0:
            all_psax_cine.append(torch.zeros_like(all_plax_cine[0]))
            num_psax = 1
            no_psax = True

        if not self.train:

            num_clips = min(
                math.ceil(max(frame_nums) / self.max_frames),
                self.max_clips,
            )

            plax_psax_comb = list(
                itertools.product(list(range(num_plax)), list(range(num_psax)))
            )

            if len(plax_psax_comb) > 6:
                plax_psax_comb = plax_psax_comb[:6]

            mask = torch.ones(
                (
                    num_clips * len(plax_psax_comb),
                    2,
                    self.max_frames,
                ),
                dtype=torch.bool,
            )

            plax_temp = list()
            psax_temp = list()

            for combination_idx in range(len(plax_psax_comb)):
                for clip_idx in range(num_clips):
                    (
                        plax_cine_temp,
                        mask[(num_clips * combination_idx) + clip_idx],
                    ) = self.pad_vid(
                        all_plax_cine[plax_psax_comb[combination_idx][0]],
                        mask[(num_clips * combination_idx) + clip_idx],
                        0,
                        clip_idx,
                    )
                    plax_temp.append(plax_cine_temp.unsqueeze(0))

                    (
                        psax_cine_temp,
                        mask[(num_clips * combination_idx) + clip_idx],
                    ) = self.pad_vid(
                        all_psax_cine[plax_psax_comb[combination_idx][1]],
                        mask[(num_clips * combination_idx) + clip_idx],
                        1,
                        clip_idx,
                    )
                    psax_temp.append(psax_cine_temp.unsqueeze(0))

            plax_cine = torch.cat(plax_temp, dim=0)
            psax_cine = torch.cat(psax_temp, dim=0)
            cine = torch.cat(
                (
                    plax_cine.unsqueeze(1).unsqueeze(-1),
                    psax_cine.unsqueeze(1).unsqueeze(-1),
                ),
                dim=1,
            )

            if no_plax:
                mask[:, 0, :] = False
            elif no_psax:
                mask[:, 1, :] = False
        else:
            mask = torch.ones((2, self.max_frames), dtype=torch.bool)

            plax_cine, mask = self.pad_vid(all_plax_cine[0], mask, 0)
            psax_cine, mask = self.pad_vid(all_psax_cine[0], mask, 1)

            cine = torch.cat(
                (
                    plax_cine.unsqueeze(0).unsqueeze(-1),
                    psax_cine.unsqueeze(0).unsqueeze(-1),
                ),
                dim=0,
            )

            if no_plax:
                mask[0, :] = False
            elif no_psax:
                mask[1, :] = False

        label = label_schemes[data_info["as_label"].iloc[0]]

        return {
            "vid": cine,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "lv_mask": torch.zeros(1),
            "ed_frame": 0,
            "ed_valid": False,
            "es_frame": 0,
            "es_valid": False,
            "class_label": torch.zeros(1),
        }

    def pad_vid(self, vid, mask, mask_idx, clip_idx=0):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]
                else:
                    vid = vid[-self.max_frames :]

        return vid, mask


class KineticsDataset(Kinetics):
    def __init__(
        self,
        dataset_path,
        mode,
        max_frames=32,
        transform=None,
        split="train",
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
        mean=0.413165,
        std=0.278993,
    ):

        super().__init__(
            root=dataset_path,
            frames_per_clip=max_frames,
            num_classes="400",
            split=split,
            transform=transform,
            step_between_clips=max_frames,
            num_workers=16,
            download=False,
        )

        self.max_frames = max_frames
        self.mean_std = mean_std
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        vid, _, label = super().__getitem__(idx)

        # Move color channel to last
        vid = vid.permute(0, 2, 3, 1).to(torch.float32) / 255

        if self.mean_std:
            return vid

        vid = (vid - self.mean) / self.std

        return {
            "vid": vid.unsqueeze(0),
            "label": label,
            "mask": torch.ones((1, self.max_frames), dtype=torch.bool),
            "lv_mask": torch.zeros(1),
            "ed_frame": 0,
            "ed_valid": False,
            "es_frame": 0,
            "es_valid": False,
            "class_label": torch.zeros(1),
        }

import os
import cv2
import glob
import torch
import numpy as np

from torch.utils.data import Dataset
from .video_augmentation import *

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class VideoDataset(Dataset):
    def __init__(
        self,
        info_dir,
        img_dir,
        num_gloss=-1,
        mode="train",
        transform_mode=True,
    ):
        self.img_dir = img_dir

        self.mode = mode
        self.ng = num_gloss
        self.gloss_dict = np.load(
            f"{info_dir}/gloss_dict.npy", allow_pickle=True
        ).item()

        self.inputs_list = np.load(
            f"{info_dir}/{mode}_info.npy", allow_pickle=True
        ).item()

        if transform_mode == "train":
            self.transform = Compose(
                [
                    RandomCrop(224),
                    RandomHorizontalFlip(0.5),
                    ToTensor(),
                    TemporalRescale(0.2),
                ]
            )
        else:
            self.transform = Compose([CenterCrop(224), ToTensor()])

    def __getitem__(self, idx):
        fi = self.inputs_list[idx]
        img_folder = os.path.join(self.img_dir, fi["folder"])
        img_list = sorted(glob.glob(img_folder))

        label_list = []
        for phase in fi["label"].split(" "):
            if phase == "":
                continue
            if phase in self.gloss_dict.keys():
                label_list.append(self.gloss_dict[phase][0])
        input_data = [
            cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            for img_path in img_list
        ]
        label = torch.LongTensor(label_list)

        input_data = self.transform(input_data)
        input_data = input_data.float() / 127.5 - 1

        return input_data, label, self.inputs_list[idx]["original_info"]

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video]
            )
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [
                torch.cat(
                    (
                        vid[0][None].expand(left_pad, -1, -1, -1),
                        vid,
                        vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [
                torch.cat(
                    (
                        vid,
                        vid[-1][None].expand(max_len - len(vid), -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    # def record_time(self):
    #     self.cur_time = time.time()
    #     return self.cur_time

    # def split_time(self):
    #     split_time = time.time() - self.cur_time
    #     self.record_time()
    #     return split_time

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .video_augmentation import *
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class VideoDataset(Dataset):
    def __init__(self, gloss_dict, annotations, img_dir, mode="train"):
        self.img_dir = img_dir

        self.mode = mode
        self.gloss_dict = gloss_dict

        self.annotations = annotations

        if mode == "train":
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
        info = self.annotations.iloc[idx]
        img_folder = Path(self.img_dir) / self.mode / info["fileid"]
        img_list = sorted([str(p) for p in list(img_folder.glob(info["pattern"]))])

        label_list = []
        for phase in info["label"].split(" "):
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

        return input_data, label, info["fileid"]

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
        return self.annotations.shape[0]


class VideoDataModule(LightningDataModule):
    def __init__(self, info_dir, img_dir, gloss_dict, batch_size, num_worker=1, **args):
        super().__init__()

        self.info_dir = info_dir
        self.img_dir = img_dir

        self.batch_size = batch_size
        self.num_worker = num_worker

        self.gloss_dict = gloss_dict
        self.datasets = {"fit": None, "validate": None, "test": None}

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage == "validate":
            if stage == "fit":
                trn_annotations = pd.read_csv(f"{self.info_dir}/train_info.csv")
                self.datasets["fit"] = VideoDataset(
                    self.gloss_dict, trn_annotations, self.img_dir, "train"
                )

            val_annotations = pd.read_csv(f"{self.info_dir}/dev_info.csv")
            self.datasets["validate"] = VideoDataset(
                self.gloss_dict, val_annotations, self.img_dir, "dev"
            )
        elif stage == "test":
            test_annotations = pd.read_csv(f"{self.info_dir}/test_info.csv")
            self.datasets["test"] = VideoDataset(
                self.gloss_dict, test_annotations, self.img_dir, "test"
            )

    def train_dataloader(self):
        dataset = self.datasets["fit"]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_worker,
            collate_fn=dataset.collate_fn,
        )

    def val_dataloader(self):
        dataset = self.datasets["validate"]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_worker,
            collate_fn=dataset.collate_fn,
        )

    def test_dataloader(self):
        dataset = self.datasets["test"]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_worker,
            collate_fn=dataset.collate_fn,
        )

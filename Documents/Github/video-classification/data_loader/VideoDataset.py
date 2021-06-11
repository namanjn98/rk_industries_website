import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader.helper import *

class VideoDataset(Dataset):
    def __init__(
        self,
        info_path,
        **kwargs,
    ):
        self._info_path = info_path
        self.load_infos(self._info_path)

    def load_infos(self, info_path):
        
        self._video_infos = open(info_path, "r").readlines()
        print("Loaded {} Objects".format(len(self._video_infos)))

    def __len__(self):
        return len(self._video_infos)

    def get_video_data(self, idx):
        info = self._video_infos[idx]
        video_path, video_label = info.strip().split(" ")
        video = videoFile_to_array(video_path)
        
        assert video != -1, "Video corrupted {}".format(video_path)
        return video, video_label

    def __getitem__(self, idx):
        return self.get_video_data(idx)
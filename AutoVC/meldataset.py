"""
References: https://github.com/yl4579/StarGANv2-VC/blob/main/meldataset.py
"""

import random
import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=22050,
                 validation=False):
        _data_list = [pair[:-1].split('|') for pair in data_list]  # *.wav|id
        self.data_list = [(path, int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))
        }  # 为每一个class建立单独的data_list

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4  # ISSUE
        self.validation = validation
        self.max_mel_length = 192  # TODO: 可以不hardcode

    # 数据条数
    def __len__(self):
        return len(self.data_list)

    # dataloader拿出的数据
    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label = self._load_data(data)

    # 根据path加载data
    def _load_data(self, data):
        wave_tensor, label = self._load_tensor(data)

        # ISSUE: necessary?
        if not self.validation:  # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()  # scale介于0.5和0.75之间
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)  # [B, L, n_mels]
        if mel_length > self.max_mel_length:  # 太长，则随机截取一段
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label

    # 将wave tensor转成mel tensor
    def _preprocess(self, wave_tensor):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std

        return mel_tensor

    # 根据path加载data，并打上label
    @staticmethod
    def _load_tensor(data):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()  # wav以int存储，但训练时要用float

        return wave_tensor, label


class MelCollate(object):
    def __init__(self, return_wave=False):
        self.return_wave = return_wave
        self.max_mel_length = 192

    def __call__(self, batch):
        batch_size = len(batch)
        # TODO: 选出batch中最长的tensor作为max_length来做padding
        n_mels = batch_size[0][0].size(0)
        mels = torch.zeros(batch_size, n_mels, self.max_mel_length).float()
        labels = torch.zeros(batch_size).long()

        for bid, (mel, label) in enumerate(batch):
            mel_size = mel.size(1)  # 长度
            mels[bid, :, mel_size] = mel

            labels[bid] = label

        mels = mels.unsqueeze(1)
        return mels, labels


def build_dataloader(data_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config=None):
    if collate_config is None:
        collate_config = {}

    dataset = MelDataset(data_list, validation=validation)
    collate_fn = MelCollate(**collate_config)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

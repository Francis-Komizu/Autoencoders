"""
References: https://github.com/jaywalnut310/vits/blob/main/utils.py
"""

import os
import glob
import sys
import logging
import json
import numpy as np
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:  # resume training
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict(checkpoint_dict['model'])
    print('Loaded model and optimizer state at epoch {} from {}'.format(epoch, checkpoint_path))

    return model, optimizer, learning_rate, epoch


def save_checkpoint(checkpoint_path, model, optimizer, learning_rate, epoch):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'epoch': epoch
    }

    torch.save(state_dict, checkpoint_path)
    print('Saved model and optimizer state at epoch {} to {}'.format(epoch, checkpoint_path))


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_data_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = 'data/train_list.txt'
    if val_path is None:
        val_path = 'data/val_list.txt'

    with open(train_path, 'r', encoding='utf-8') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_list = f.readlines()

    return train_list, val_list


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

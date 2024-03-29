from typing import Optional

import numpy as np

import torch
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig

from deepy.nn.model import VGG2d


def is_same_config(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    """Compare cfg1 with cfg2.

    Args:
        cfg1: Config
        cfg2: Config

    Returns:
        True if cfg1 == cfg2 else False

    """
    return cfg1 == cfg2


def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(OmegaConf.to_yaml(cfg))
    print('--------------------')


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(classes: list[str], cfg: DictConfig) -> nn.Module:
    if cfg.model.name == "vgg2d":
        net = VGG2d(
            in_channels=1,
            num_classes=len(classes),
            base_channels=cfg.model.param.base_channels,
            depth=cfg.model.param.depth,
            max_channels=cfg.model.param.max_channels,
            conv=nn.Conv2d,
            down_conv=nn.Conv2d,
            activation=nn.ReLU)
    else:
        raise NotImplementedError()

    return net

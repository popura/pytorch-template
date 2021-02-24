import typing
from typing import List
import torch
import torch.nn as nn
from torchsummary import summary

from omegaconf import DictConfig

import model as mymodel


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
    print(cfg.pretty())
    print('--------------------')


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(classes: List[str], cfg: DictConfig) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.name == "baseline_cnn":
        net = mymodel.BaselineCNN(len(classes), cfg.dataset.transforms.mel_spectrogram.n_mels)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(1, 40, 501))
    elif cfg.model.name == "vgg1d":
        net = mymodel.VGG1d(in_channels=1, num_classes=len(classes),
                            base_channels=cfg.model.param.base_channels,
                            depth=cfg.model.param.depth,
                            max_channels=cfg.model.param.max_channels,
                            down_conv_layer=nn.Conv1d,
                            activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(1, 16000*10))
    elif cfg.model.name == "multi_vgg1d":
        net = mymodel.MultiVGG1d(in_channels=1,
                                 num_inputs=cfg.model.param.num_inputs,
                                 num_classes=len(classes),
                                 base_channels=cfg.model.param.base_channels,
                                 depth=cfg.model.param.depth,
                                 max_channels=cfg.model.param.max_channels,
                                 down_conv_layer=nn.Conv1d,
                                 activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
    elif cfg.model.name == "vgg2d":
        net = mymodel.VGG2d(in_channels=1,
                            num_classes=len(classes),
                            base_channels=cfg.model.param.base_channels,
                            depth=cfg.model.param.depth,
                            max_channels=cfg.model.param.max_channels,
                            down_conv_layer=nn.Conv2d,
                            activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
    elif cfg.model.name == "multi_vgg2d":
        net = mymodel.MultiVGG2d(in_channels=1,
                                 num_inputs=cfg.model.param.num_inputs,
                                 num_classes=len(classes),
                                 base_channels=cfg.model.param.base_channels,
                                 depth=cfg.model.param.depth,
                                 max_channels=cfg.model.param.max_channels,
                                 down_conv_layer=nn.Conv2d,
                                 activation=nn.ReLU)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
    elif cfg.model.name == "blinky_vgg1d":
        net = mymodel.BlinkyVGG1d(in_channels=1, num_classes=len(classes),
                                  base_channels=cfg.model.param.base_channels,
                                  sample_rate=cfg.dataset.sample_rate,
                                  depth=cfg.model.param.depth,
                                  max_channels=cfg.model.param.max_channels,
                                  down_conv_layer=nn.Conv1d,
                                  activation=nn.ReLU,
                                  audio_buffer_size=cfg.model.param.audio_buffer_size,
                                  frame_rate=cfg.model.param.frame_rate,
                                  temperature=cfg.model.param.temperature)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(1, 16000*10))
    elif cfg.model.name == "multi_blinky_vgg1d":
        net = mymodel.MultiBlinkyVGG1d(in_channels=1,
                                       num_inputs=cfg.model.param.num_inputs,
                                       num_classes=len(classes),
                                       base_channels=cfg.model.param.base_channels,
                                       sample_rate=cfg.dataset.sample_rate,
                                       distances=list(cfg.model.param.distances),
                                       biases=list(cfg.model.param.biases),
                                       stds=list(cfg.model.param.stds),
                                       depth=cfg.model.param.depth,
                                       max_channels=cfg.model.param.max_channels,
                                       down_conv_layer=nn.Conv1d,
                                       activation=nn.ReLU,
                                       audio_buffer_size=cfg.model.param.audio_buffer_size,
                                       frame_rate=cfg.model.param.frame_rate,
                                       temperature=cfg.model.param.temperature)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        #summary(net, input_size=(1, 16000*10))
    elif cfg.model.name == "unet_vgg1d":
        net = mymodel.UNetVGG1d(in_channels=1, num_classes=len(classes),
                                base_channels=cfg.model.param.base_channels,
                                sample_rate=cfg.dataset.sample_rate,
                                depth=cfg.model.param.depth,
                                max_channels=cfg.model.param.max_channels,
                                up_conv_layer=nn.ConvTranspose1d,
                                down_conv_layer=nn.Conv1d,
                                activation=nn.ReLU,
                                final_activation=nn.Identity,
                                audio_buffer_size=cfg.model.param.audio_buffer_size,
                                frame_rate=cfg.model.param.frame_rate,
                                temperature=cfg.model.param.temperature)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        summary(net, input_size=(1, 16000*10))
    elif cfg.model.name == "multi_unet_vgg1d":
        net = mymodel.MultiUNetVGG1d(in_channels=1,
                                     num_inputs=cfg.model.param.num_inputs,
                                     num_classes=len(classes),
                                     base_channels=cfg.model.param.base_channels,
                                     sample_rate=cfg.dataset.sample_rate,
                                     distances=list(cfg.model.param.distances),
                                     biases=list(cfg.model.param.biases),
                                     stds=list(cfg.model.param.stds),
                                     depth=cfg.model.param.depth,
                                     max_channels=cfg.model.param.max_channels,
                                     up_conv_layer=nn.ConvTranspose1d,
                                     down_conv_layer=nn.Conv1d,
                                     activation=nn.ReLU,
                                     final_activation=nn.Identity,
                                     audio_buffer_size=cfg.model.param.audio_buffer_size,
                                     frame_rate=cfg.model.param.frame_rate,
                                     temperature=cfg.model.param.temperature)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        #summary(net, input_size=(1, 16000*10))
    elif cfg.model.name == "multi_encoder_vgg1d":
        net = mymodel.MultiEncoderVGG1d(in_channels=1,
                                        num_inputs=cfg.model.param.num_inputs,
                                        num_classes=len(classes),
                                        base_channels=cfg.model.param.base_channels,
                                        sample_rate=cfg.dataset.sample_rate,
                                        distances=list(cfg.model.param.distances),
                                        biases=list(cfg.model.param.biases),
                                        stds=list(cfg.model.param.stds),
                                        depth=cfg.model.param.depth,
                                        max_channels=cfg.model.param.max_channels,
                                        down_conv_layer=nn.Conv1d,
                                        activation=nn.ReLU,
                                        final_activation=nn.Identity,
                                        audio_buffer_size=cfg.model.param.audio_buffer_size,
                                        frame_rate=cfg.model.param.frame_rate,
                                        temperature=cfg.model.param.temperature)
        if device.type == "cuda":
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        #summary(net, input_size=(1, 16000*10))
    else:
        raise NotImplementedError()

    return net

import time
import argparse
from pathlib import Path
import hashlib
import typing

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchsummary import summary

from omegaconf import DictConfig, OmegaConf
import hydra

import deepy.data.transform
import deepy.data.audio.transform
from deepy.train.trainer import (
    ClassifierTrainer,
    MultiInputClassifierTrainer
)
from deepy.train.extension import (
    IntervalTrigger,
    MinValueTrigger,
    MaxValueTrigger,
    ModelSaver,
    HistorySaver
)

import model as mymodel
import dataset as mydataset
from util import get_model


DATA_MODES = ('dev_data', 'eval_data')


def save_model(model: nn.Module, path: str) -> typing.NoReturn:
    """Save a DNN model (torch.nn.Module).

    Args:
        model: torch.nn.Module object
        path: Directory path where model will be saved

    Returns:

    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), path)



def get_transforms(cfg: DictConfig, generator=None):
    if cfg.model.name in ["baseline_cnn", "vgg2d", "multi_vgg2d"]:
        transforms = deepy.data.transform.Compose(
            [deepy.data.audio.transform.RandomCrop(**cfg.dataset.transforms.random_crop,
                                                   generator=generator),
             deepy.data.transform.Lambda(lambda x: torch.mean(x, dim=0, keepdim=True)),
             torchaudio.transforms.MelSpectrogram(**cfg.dataset.transforms.mel_spectrogram),
             torchaudio.transforms.AmplitudeToDB(**cfg.dataset.transforms.amplitude_to_db)])
    elif cfg.model.name in ["vgg1d", "multi_vgg1d", "blinky_vgg1d", "multi_blinky_vgg1d",
                            "unet_vgg1d", "multi_unet_vgg1d", "multi_encoder_vgg1d"]:
        transforms = deepy.data.transform.Compose(
            [deepy.data.audio.transform.RandomCrop(**cfg.dataset.transforms.random_crop,
                                                   generator=generator),
             deepy.data.transform.Lambda(lambda x: torch.mean(x, dim=0, keepdim=True))])
    else:
        raise NotImplementedError()
    return transforms


def get_data_loaders(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    p = Path(cwd) / cfg.dataset.path

    
    nodes = cfg.dataset.nodes
    if "multi" in cfg.model.name:
        trainsets = []
        valsets = []
        for i in nodes:
            transforms = get_transforms(
                cfg=cfg,
                generator=torch.Generator().manual_seed(2380314)
            )

            trainsets.append(
                mydataset.DCASE2018Task5Dataset(str(p),
                                                mode='train',
                                                node=[i],
                                                pre_load=cfg.dataset.pre_load,
                                                transform=transforms)
            )
            valsets.append(
                mydataset.DCASE2018Task5Dataset(str(p),
                                                mode='evaluate',
                                                node=[i],
                                                pre_load=cfg.dataset.pre_load,
                                                transform=transforms)
            )

        classes = trainsets[0].classes

        generators = [torch.Generator() for i in range(len(trainsets))]
        for g in generators:
            g = g.manual_seed(13459870123)
        samplers = [torch.utils.data.RandomSampler(ds, generator=g)
                    for ds, g in zip(trainsets, generators)]
        trainloaders = [torch.utils.data.DataLoader(ds,
                                                    batch_size=cfg.loader.batch_size,
                                                    sampler=s,
                                                    num_workers=cfg.loader.num_workers)
                        for ds, s in zip(trainsets, samplers)]
        valloaders = [torch.utils.data.DataLoader(ds, batch_size=cfg.loader.batch_size,
                                                  shuffle=False, num_workers=cfg.loader.num_workers)
                      for ds in trainsets]
    else:
        transforms = get_transforms(
            cfg=cfg,
            generator=torch.Generator().manual_seed(2380314)
        )
        trainsets = mydataset.DCASE2018Task5Dataset(str(p),
                                                    mode='train',
                                                    pre_load=cfg.dataset.pre_load,
                                                    transform=transforms)
        valsets = mydataset.DCASE2018Task5Dataset(str(p),
                                                  mode='evaluate',
                                                  pre_load=cfg.dataset.pre_load,
                                                  transform=transforms)

        classes = trainsets.classes

        trainloaders = torch.utils.data.DataLoader(trainsets,
                                                   batch_size=cfg.loader.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.loader.num_workers)
        valloaders = torch.utils.data.DataLoader(valsets, batch_size=cfg.loader.batch_size,
                                                 shuffle=False, num_workers=cfg.loader.num_workers)
    
    return trainloaders, valloaders, classes


def show_history(train_loss, val_loss, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss,
             label='Accuracy for training data')
    plt.plot(range(len(val_loss)), val_loss,
             label='Accuracy for val data')
    plt.legend()
    plt.savefig(path)


def get_optimizer(params, name: str, **hyper_params):
    if name == 'sgd':
        return optim.SGD(params, **hyper_params)
    elif name == 'adam':
        return optim.Adam(params, **hyper_params)
    else:
        raise NotImplementedError
    return


def get_scheduler(optimizer, cfg):
    if cfg.lr_scheduler.name == 'multi_step':
        return optim.lr_scheduler.MultiStepLR(optimizer, **cfg.lr_scheduler.params)
    elif cfg.lr_scheduler.name is None:
        return None
    else:
        raise NotImplementedError
    return


@hydra.main(config_path='../conf/config.yaml')
def main(cfg: DictConfig) -> None:
    cwd = Path(hydra.utils.get_original_cwd())

    # Printing cfg
    print_config(cfg)

    # Setting history directory
    # All outputs will be written into (p / 'history' / train_id).
    if not (cwd / 'history').exists():
        (cwd / 'history').mkdir(parents=True)
    train_id = generate_train_id(cfg)
    p = cwd / 'history' / train_id
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

    cfg_path = p / 'config.yaml'
    if cfg_path.exists():
        existing_cfg = OmegaConf.load(str(p / 'config.yaml'))
        if not is_same_config(cfg, existing_cfg):
            raise ValueError("Train ID {} already exists, but config is different".format(train_id))

    # Saving cfg
    OmegaConf.save(cfg, str(p / 'config.yaml'))

    # Setting seed 
    if cfg.seed is not None:
        myutil.set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    trainloaders, valloaders, classes = get_data_loaders(cfg)
    net = get_model(classes, cfg)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net.parameters(),
                              cfg.optimizer.name,
                              **cfg.optimizer.params)
    scheduler = get_scheduler(optimizer, cfg)
    extensions = [ModelSaver(directory=p,
                             name=lambda x: cfg.model.name+"_best.pth",
                             trigger=MaxValueTrigger(mode="validation", key="total acc")),
                  HistorySaver(directory=p,
                               name=lambda x: cfg.model.name+"_history.pth",
                               trigger=IntervalTrigger(period=1))]

    if "multi" in cfg.model.name:
        if cfg.dataset.shuffle_nodes:
            trainer_cls = MultiShuffledInputClassifierTrainer
        else:
            trainer_cls = MultiInputClassifierTrainer
        trainer = trainer_cls(net, optimizer, criterion, trainloaders,
                              scheduler=scheduler, extensions=extensions,
                              init_epoch=0,
                              device=device)
    else:
        trainer = ClassifierTrainer(net, optimizer, criterion, trainloaders,
                                    scheduler=scheduler, extensions=extensions,
                                    init_epoch=0,
                                    device=device)
    
    try:
        trainer.train(cfg.epoch, valloaders, classes)
    except:
        raise

    save_model(net, str(p / '{}.pth'.format(cfg.model.name)))


if __name__ == "__main__":
    main()

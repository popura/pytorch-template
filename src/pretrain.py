import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchsummary import summary

from omegaconf import DictConfig
import hydra

import deepy.data.transform
import deepy.data.audio.transform
from deepy.train.trainer import RegressorTrainer

import model as mymodel
import dataset as mydataset
from trainer import MultiInputClassifierTrainer
from util import get_model


DATA_MODES = ('dev_data', 'eval_data')

def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(cfg.pretty())
    print('--------------------')

def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    p = Path(cwd) / cfg.dataset.path

    if cfg.model.name == "baseline_cnn":
        pre_transforms = deepy.data.transform.Compose(
            [deepy.data.audio.transform.RandomCrop(**cfg.dataset.transforms.random_crop),
             deepy.data.transform.Lambda(lambda x: torch.mean(x, dim=0, keepdim=True)),
             torchaudio.transforms.MelSpectrogram(**cfg.dataset.transforms.mel_spectrogram),
             torchaudio.transforms.AmplitudeToDB(**cfg.dataset.transforms.amplitude_to_db)])
    elif cfg.model.name in ["vgg1d", "multi_vgg1d", "blinky_vgg1d", "multi_blinky_vgg1d", "unet_vgg1d", "multi_unet_vgg1d"]:
        pre_transforms = deepy.data.transform.Compose(
            [deepy.data.audio.transform.RandomCrop(**cfg.dataset.transforms.random_crop),
             deepy.data.transform.Lambda(lambda x: torch.mean(x, dim=0, keepdim=True))])
    else:
        raise NotImplementedError()
    
    if "multi" in cfg.model.name:
        trainsets = [mydataset.DCASE2018Task5Dataset(str(p),
                                                     mode='train',
                                                     node=[i],
                                                     pre_load=True,
                                                     pre_transform=pre_transforms)
                     for i in range(2, 5)]
        valsets = [mydataset.DCASE2018Task5Dataset(str(p),
                                                   mode='evaluate',
                                                   node=[i],
                                                   pre_load=True,
                                                   pre_transform=pre_transforms)
                   for i in range(2, 5)]

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
        trainsets = mydataset.DCASE2018Task5Dataset(str(p),
                                                    mode='train',
                                                    pre_load=True,
                                                    pre_transform=pre_transforms)
        valsets = mydataset.DCASE2018Task5Dataset(str(p),
                                                  mode='evaluate',
                                                  pre_load=True,
                                                  pre_transform=pre_transforms)

        classes = trainsets.classes

        trainloaders = torch.utils.data.DataLoader(trainsets,
                                                   batch_size=cfg.loader.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.loader.num_workers)
        valloaders = torch.utils.data.DataLoader(valsets, batch_size=cfg.loader.batch_size,
                                                 shuffle=False, num_workers=cfg.loader.num_workers)
    
    return trainloaders, valloaders, classes

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

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
    print_config(cfg)
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloaders, valloaders, classes = get_data_loaders(cfg)
    net = UNet1d(in_channels=in_channels,
                           out_channels=in_channels,
                           base_channels=base_channels,
                           depth=depth,
                           max_channels=max_channels,
                           up_conv_layer=up_conv_layer,
                           down_conv_layer=down_conv_layer,
                           activation=activation,
                           final_activation=final_activation)
    net = net.to(device)
    if device.type == "cuda":
        net = torch.nn.DataParallel(net)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net.parameters(),
                              cfg.optimizer.name,
                              **cfg.optimizer.params)
    scheduler = get_scheduler(optimizer, cfg)

    if "multi" in cfg.model.name:
        trainer = MultiInputClassifierTrainer(net, optimizer, criterion, trainloaders,
                                              scheduler=scheduler, init_epoch=0,
                                              device=device)
    else:
        trainer = ClassifierTrainer(net, optimizer, criterion, trainloaders,
                                    scheduler=scheduler, init_epoch=0,
                                    device=device)
    trainer.train(cfg.epoch, valloaders, classes)

    cwd = hydra.utils.get_original_cwd()
    p = Path(cwd)
    save_model(net, str(p / cfg.model.path / '{}.pth'.format(cfg.model.name)))
    #show_history(train_accs, val_accs, str(p / 'history' / '{}_history.png'.format(cfg.model.name)))


if __name__ == "__main__":
    main()

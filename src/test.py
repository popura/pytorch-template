import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchsummary import summary

from omegaconf import OmegaConf, DictConfig
import hydra

import deepy.data.transform
from deepy.train.trainer import ClassifierTrainer

import model as mymodel
import dataset as mydataset
from trainer import MultiInputClassifierTrainer
from util import get_model



def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(cfg.pretty())
    print('--------------------')


def get_data_loader(cfg: DictConfig):
    cwd = Path.cwd()
    p = cwd / cfg.dataset.path

    if cfg.model.name in ["baseline_cnn", "vgg2d", "multi_vgg2d"]:
        pre_transforms = deepy.data.transform.Compose(
            [deepy.data.transform.Lambda(lambda x: torch.mean(x, dim=0, keepdim=True)),
             torchaudio.transforms.MelSpectrogram(**cfg.dataset.transforms.mel_spectrogram),
             torchaudio.transforms.AmplitudeToDB(**cfg.dataset.transforms.amplitude_to_db)])
    elif cfg.model.name in ["vgg1d", "multi_vgg1d", "blinky_vgg1d", "multi_blinky_vgg1d",
                            "unet_vgg1d", "multi_unet_vgg1d", "multi_encoder_vgg1d"]:
        pre_transforms = deepy.data.transform.Compose(
            [deepy.data.transform.Lambda(lambda x: torch.mean(x, dim=0, keepdim=True))])
    else:
        raise NotImplementedError()

    nodes = cfg.dataset.nodes
    if "multi" in cfg.model.name:
        testsets = [mydataset.DCASE2018Task5Dataset(str(p),
                                                    mode='test',
                                                    node=[i],
                                                    pre_load=True,
                                                    pre_transform=pre_transforms)
                    for i in nodes]
        classes = testsets[0].classes
        testloaders = [torch.utils.data.DataLoader(ds, batch_size=cfg.loader.batch_size,
                                                   shuffle=False, num_workers=cfg.loader.num_workers)
                       for ds in testsets]
    else:
        testsets = mydataset.DCASE2018Task5Dataset(str(p),
                                                   mode='test',
                                                   node=nodes,
                                                   pre_load=True,
                                                   pre_transform=pre_transforms)
        classes = testsets.classes
        testloaders = torch.utils.data.DataLoader(testsets, batch_size=cfg.loader.batch_size,
                                                  shuffle=False, num_workers=cfg.loader.num_workers)
    
    return testloaders, classes


def examples(path, net, datasets, classes, device, cfg):
    if cfg.model.name == "multi_blinky_vgg1d":
        net.eval()
        with torch.no_grad():
            inputs = []
            labels = []
            for j, ds in enumerate(datasets):
                x = []
                tmp_labels = []
                for cls_ in classes:
                    cls_idx = ds.class_to_idx[cls_]
                    sample_idx = ds.targets.index(cls_idx)

                    tmp_input, tmp_label = ds[sample_idx]
                    tmp_input = tmp_input.to(device)
                    x.append(tmp_input)
                    tmp_labels.append(tmp_label)

                inputs.append(torch.stack(x, dim=0))
                if j == 0:
                    labels = torch.tensor(tmp_labels).to(device)
                else:
                    if not (labels == torch.tensor(tmp_labels).to(device)).all():
                        raise ValueError('Different labels are loaded')

            y = []
            z = []
            for j, x in enumerate(inputs):
                tmp_y = net.blinkies[j](x) 
                y.append(tmp_y.clone().detach().cpu())
                tmp_y = net.lights[j](tmp_y)
                tmp_y = net.camera(tmp_y)
                z.append(tmp_y.clone().detach().cpu())
    elif cfg.model.name == "multi_unet_vgg1d":
        net.eval()
        with torch.no_grad():
            inputs = []
            labels = []
            for j, ds in enumerate(datasets):
                x = []
                tmp_labels = []
                for cls_ in classes:
                    cls_idx = ds.class_to_idx[cls_]
                    sample_idx = ds.targets.index(cls_idx)

                    tmp_input, tmp_label = ds[sample_idx]
                    tmp_input = tmp_input.to(device)
                    x.append(tmp_input)
                    tmp_labels.append(tmp_label)

                inputs.append(torch.stack(x, dim=0))
                if j == 0:
                    labels = torch.tensor(tmp_labels).to(device)
                else:
                    if not (labels == torch.tensor(tmp_labels).to(device)).all():
                        raise ValueError('Different labels are loaded')

            y = []
            z = []
            for j, x in enumerate(inputs):
                tmp_y = net.unets[j](x) 
                tmp_y = net.normalization(tmp_y)
                y.append(tmp_y.clone().detach().cpu())
                tmp_y = net.lights[j](tmp_y)
                tmp_y = net.camera(tmp_y)
                z.append(tmp_y.clone().detach().cpu())
    elif cfg.model.name == "multi_encoder_vgg1d":
        net.eval()
        with torch.no_grad():
            inputs = []
            labels = []
            for j, ds in enumerate(datasets):
                x = []
                tmp_labels = []
                for cls_ in classes:
                    cls_idx = ds.class_to_idx[cls_]
                    sample_idx = ds.targets.index(cls_idx)

                    tmp_input, tmp_label = ds[sample_idx]
                    tmp_input = tmp_input.to(device)
                    x.append(tmp_input)
                    tmp_labels.append(tmp_label)

                inputs.append(torch.stack(x, dim=0))
                if j == 0:
                    labels = torch.tensor(tmp_labels).to(device)
                else:
                    if not (labels == torch.tensor(tmp_labels).to(device)).all():
                        raise ValueError('Different labels are loaded')

            y = []
            z = []
            for j, x in enumerate(inputs):
                tmp_y = net.unets[j](x) 
                tmp_y = net.normalization(tmp_y)
                y.append(tmp_y.clone().detach().cpu())
                tmp_y = net.lights[j](tmp_y)
                tmp_y = net.camera(tmp_y)
                z.append(tmp_y.clone().detach().cpu())
    else:
        return
    
    for i, (org, sound, pixel) in enumerate(zip(inputs, y, z)):
        org = org.clone().detach().cpu()
        for j in range(org.shape[0]):
            plt.figure()
            plt.plot(org[j, 0, :])
            plt.savefig(str(path / '{}_node{}_org.png'.format(classes[j], i+1)))
            plt.clf()
            plt.close()
            torchaudio.save(str(path / '{}_node{}_org.wav'.format(classes[j], i+1)), org[j], cfg.dataset.sample_rate)

            plt.figure()
            plt.plot(sound[j, 0, :])
            plt.savefig(str(path / '{}_node{}_sound.png'.format(classes[j], i+1)))
            plt.clf()
            plt.close()
            torchaudio.save(str(path / '{}_node{}_sound.wav'.format(classes[j], i+1)), sound[j], cfg.dataset.sample_rate)

            plt.figure()
            plt.plot(pixel[j, 0, :])
            plt.savefig(str(path / '{}_node{}_pixel.png'.format(classes[j], i+1)))
            plt.clf()
            plt.close()
    
    z = torch.cat(z, dim=1)
    for j in range(z.shape[0]):
        plt.figure(figsize=(10.0, 5.0))
        plt.imshow(z[j], interpolation='nearest',vmin=0,vmax=1,cmap='jet', aspect=4.0)
        plt.savefig(str(path / '{}_emitted_feature_map.pdf'.format(classes[j])))
        plt.clf()
        plt.close()

    return


def calculate_confusion_matrix(net, dataloaders, classes, device):
    net.eval()
    true_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    pred_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]
    with torch.no_grad():
        dl_iters = [dataloaders[i].__iter__()
                    for i in range(len(dataloaders))]
        for i in range(len(dataloaders[0])):
            inputs = []
            for j in range(len(dataloaders)):
                tmp_inputs, tmp_labels = dl_iters[j].next()
                tmp_inputs = tmp_inputs.to(device)
                tmp_labels = tmp_labels.to(device)
                if j == 0:
                    labels = tmp_labels
                else:
                    if not (labels == tmp_labels).all():
                        raise ValueError('Different labels are loaded')
                inputs.append(tmp_inputs)

            outputs = net(*inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.cat([true_labels, labels.clone().detach().view(-1).cpu()])
            pred_labels = torch.cat([pred_labels, predicted.clone().detach().view(-1).cpu()])

    conf_mat = confusion_matrix(true_labels.numpy(), pred_labels.numpy())
    return conf_mat


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(cfg: DictConfig, train_id:str) -> None:
    cwd = Path.cwd()
    print_config(cfg)
    set_random_seed(0)

    model_file_name = "{}_best.pth".format(cfg.model.name)
    
    # Checking history directory
    history_dir = cwd / 'history' / train_id
    if (history_dir / model_file_name).exists():
        pass
    else:
        return

    # Setting result directory
    # All outputs will be written into (p / 'result' / train_id).
    if not (cwd / 'result').exists():
        (cwd / 'result').mkdir(parents=True)
    result_dir = cwd / 'result' / train_id
    if result_dir.exists():
        # removing files in result_dir?
        pass
    else:
        result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloaders, classes = get_data_loader(cfg)
    net = get_model(classes, cfg)

    net.module.load_state_dict(torch.load(str(history_dir / model_file_name)))
    net.to(device)
    net.eval()

    criterion = None
    optimizer = None
    scheduler = None

    if "multi" in cfg.model.name:
        trainer = MultiInputClassifierTrainer(net, optimizer, criterion, testloaders,
                                              scheduler=scheduler, init_epoch=0,
                                              device=device)
    else:
        trainer = ClassifierTrainer(net, optimizer, criterion, testloaders,
                                    scheduler=scheduler, init_epoch=0,
                                    device=device)

    history = trainer.eval(testloaders, classes)
    df = pd.DataFrame.from_dict(history, orient="index")
    df.to_csv(str(result_dir / "accuracy.csv"))
    matrix = calculate_confusion_matrix(net, testloaders, classes, device)
    print(history)
    print(matrix)

    cmd = ConfusionMatrixDisplay(matrix, display_labels=classes)
    cmd.plot(xticks_rotation='vertical')
    plt.savefig(str(result_dir / "confusion_matrix.pdf"))

    if "multi" in cfg.model.name:
        examples(result_dir, net.module, [dl.dataset for dl in testloaders], classes, device, cfg)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default='./history',
                        help='Directory path for searching trained models')
    args = parser.parse_args()
    p = Path.cwd() / args.history_dir
    for q in p.glob('**/config.yaml'):
        cfg = OmegaConf.load(str(q))
        cfg.loader.num_workers = 0
        train_id = q.parent.name
        main(cfg, train_id)

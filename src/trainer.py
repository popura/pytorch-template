import time
import os
from abc import ABCMeta
from abc import abstractmethod
import typing
import random

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from deepy.train.trainer import MultiInputClassifierTrainer


class MultiShuffledInputClassifierTrainer(MultiInputClassifierTrainer):
    """

    Args:
        net: an nn.Module. It should be applicable for multiple inputs
        dataloaders: len(dataloaders) should be equal to the number of inputs
                     and len(dataloader) for each dataloader should be the same as each other.
    """
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloaders,
                 scheduler=None,
                 extensions=None,
                 init_epoch=0,
                 device='cpu'):
        super().__init__(
            net, optimizer, criterion, dataloaders,
            scheduler=scheduler, extensions=extensions,
            init_epoch=init_epoch, device=device)

    def step(self):
        self.net.train()
        loss_meter = AverageMeter()
        dl_iters = [self.dataloaders[i].__iter__()
                    for i in range(len(self.dataloaders))]
        for i in range(len(self.dataloaders[0])):
            inputs = []
            for j in range(len(self.dataloaders)):
                tmp_inputs, tmp_labels = dl_iters[j].next()
                tmp_inputs = tmp_inputs.to(self.device)
                tmp_labels = tmp_labels.to(self.device)
                if j == 0:
                    labels = tmp_labels
                else:
                    if not (labels == tmp_labels).all():
                        raise ValueError('Different labels are loaded')
                inputs.append(tmp_inputs)
            random.shuffle(inputs)

            outputs = self.net(*inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs[0].size(0))

        if self.scheduler is not None:
            self.scheduler.step()
        self.epoch += 1
        ave_loss= loss_meter.average

        return ave_loss

import time
import os
from abc import ABC
from abc import abstractmethod
import typing

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F



class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class ABCTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    @abstractmethod
    def eval(self):
        raise NotImplementedError()

    @abstractmethod
    def extend(self):
        raise NotImplementedError()


class Trainer(ABCTrainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 scheduler=None,
                 extensions=None,
                 evaluators=None,
                 init_epoch=0,
                 device='cpu'):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.extensions = extensions
        self.evaluators = evaluators
        self.device = device
        self.epoch = init_epoch
        self.total_epoch = None
        self.history = {}

    def train(self, epochs, val_loader=None):
        start_time = time.time()
        start_epoch = self.epoch
        self.total_epoch = epochs
        self.history["train"] = []
        self.history["validation"] = []

        print('-----Training Started-----')
        self.train_setup()
        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            # loss is a scalar and self.epoch is incremented in the step function
            # (i.e. self.epoch = epoch + 1)
            loss = self.step()
            self.history["train"].append({'epoch':self.epoch, 'loss':loss})

            # vallosses is a dictionary {str: value}
            if val_loader is not None:
                vallosses = self.eval(val_loader)
                self.history["validation"].append({'epoch':self.epoch, **vallosses})

            self.extend()

        self.train_cleanup()
        print('-----Training Finished-----')

        return self.net

    def step(self):
        self.net.train()
        loss_meter = AverageMeter()
        for inputs, targets in self.dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs.size(0))

        if self.scheduler is not None:
            self.scheduler.step()
        self.epoch += 1
        ave_loss = loss_meter.average

        return ave_loss

    def eval(self, val_loader=None):
        if self.evaluators is None:
            return

        self.net.eval()
        for evaluator in self.evaluators:
            evaluator.initialize()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.net(inputs)

                for evaluator in self.evaluators:
                    evaluator.eval_batch(outputs, targets)

        hist_dict = dict()
        for evaluator in self.evaluators:
            result = evaluator.finalize()
            hist_dict = dict(**hist_dict, **result)

        return hist_dict

    def extend(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            if extension.trigger(self):
                extension(self)
        return
    
    def train_setup(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            extension.initialize(self)
        return
    
    def train_cleanup(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            extension.finalize(self)
        return


class Evaluator(ABC):
    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def eval_batch(self) -> typing.NoReturn:
        raise NotImplementedError()
    
    @abstractmethod
    def finalize(self) -> dict:
        raise NotImplementedError()


class LossEvaluator(Evaluator):
    def __init__(self, criterion, criterion_name="loss"):
        super().__init__()
        self.loss_meter = None
        self.criterion = criterion
        self.criterion_name = criterion_name

    def initialize(self):
        self.loss_meter = AverageMeter()

    def eval_batch(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.loss_meter.update(loss.item(), number=outputs.size(0))

    def finalize(self):
        return {"loss": self.loss_meter.average}


class AccuracyEvaluator(Evaluator):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.class_correct = None
        self.class_total = None
    
    def initialize(self):
        self.class_correct = list(0. for i in range(len(self.classes)))
        self.class_total = list(0. for i in range(len(self.classes)))

    def eval_batch(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets)

        for i in range(len(targets)):
            label = targets[i]
            self.class_correct[label] += c[i].item()
            self.class_total[label] += 1

    def finalize(self):
        class_accuracy = [c / t for c, t in zip(self.class_correct, self.class_total)]
        total_accuracy = sum(self.class_correct) / sum(self.class_total)

        hist_dict = {'total acc': total_accuracy}
        hist_dict = dict(**hist_dict, **{str(self.classes[i]): class_accuracy[i] for i in range(len(self.classes))})
        return hist_dict

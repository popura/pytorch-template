from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2 
import torchinfo

from pytorch_template.data_pipeline import DataPipeline
from pytorch_template.model import SimpleCNN
from pytorch_template.trainer import Trainer, LossEvaluator, AccuracyEvaluator
from pytorch_template.train_id import print_config, generate_train_id, is_same_config
from pytorch_template.extension import ModelSaver, HistorySaver, HistoryLogger, MaxValueTrigger, IntervalTrigger, LearningCurvePlotter
from pytorch_template.util import set_random_seed



if __name__ == "__main__":
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    datapipe = DataPipeline(dataset, static_transforms=transforms, dynamic_transforms=None, max_cache_size=len(dataset))
    means = torch.zeros(3)
    stds = torch.zeros(3)
    for img, _ in datapipe:
        means += img.mean(dim=[1, 2])
        stds += img.std(dim=[1, 2])
    means /= len(datapipe)
    stds /= len(datapipe)

    print(f"means: {means}, stds: {stds}")
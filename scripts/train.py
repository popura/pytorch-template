from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2 
import torchinfo

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_pipeline import DataPipeline
from src.model import SimpleCNN, ResNet
from src.trainer import Trainer, LossEvaluator, AccuracyEvaluator
from src.train_id import print_config, generate_train_id, is_same_config
from src.extension import ModelSaver, HistorySaver, HistoryLogger, MaxValueTrigger, IntervalTrigger, LearningCurvePlotter
from src.util import set_random_seed


@hydra.main(version_base=None, config_path=f"/{os.environ['PROJECT_NAME']}/conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # 乱数を固定
    set_random_seed(42)

    # 計算デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習結果の保存先ディレクトリを作る
    output_dir = Path(f"/{os.environ['PROJECT_NAME']}/outputs/{Path(__file__).stem}")
    train_id = generate_train_id(cfg)
    p = output_dir / "history" / train_id
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

    # 設定ファイルの内容を保存
    print_config(cfg)
    cfg_path = p / 'config.yaml'
    if cfg_path.exists():
        existing_cfg = OmegaConf.load(str(p / 'config.yaml'))
        if not is_same_config(cfg, existing_cfg):
            raise ValueError("Train ID {} already exists, but config is different".format(train_id))
    OmegaConf.save(cfg, str(cfg_path))
    
    # 学習・検証データセットのインスタンスを作る
    transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(**cfg.dataset.train.transform.random_resized_crop),
        v2.RandomHorizontalFlip(**cfg.dataset.train.transform.random_horizontal_flip),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])
    random_transforms = v2.Compose([
        v2.RandomResizedCrop(**cfg.dataset.train.transform.random_resized_crop),
        v2.RandomHorizontalFlip(**cfg.dataset.train.transform.random_horizontal_flip),
    ])
    dataset = torchvision.datasets.CIFAR10(
        **cfg.dataset.train.params
    )
    datapipe = DataPipeline(dataset, static_transforms=transforms, dynamic_transforms=random_transforms, max_cache_size=len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        datapipe,
        cfg.dataset.random_split.lengths
    )
    classes = dataset.classes
    
    # 学習・検証データセットを読み込むバッチを作るクラス（DataLoaderクラス）を作る
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        **cfg.dataloader
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        **cfg.dataloader
    )

    # DNNを作る
    #net = SimpleCNN(**cfg.model.params)
    net = ResNet('ResNet18').to(device)
    
    # ネットワークの構造やパラメータ数，必要なメモリ量などを表示
    input_size = train_set[0][0].shape
    torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))

    # 損失関数やその他ハイパーパラメータの定義
    epochs = cfg.epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), **cfg.optimizer.params)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [i * epochs for i in cfg.lr_scheduler.params.milestones], gamma=cfg.lr_scheduler.params.gamma)

    # 検証データセットに対する評価方法の設定
    evaluators = [
        LossEvaluator(criterion, criterion_name="cross entropy"),
        AccuracyEvaluator(classes)
    ]

    # Extension (エポック毎に実行したい処理）の設定
    extensions = [
        ModelSaver(
            directory=p,
            name=lambda x: "best_model.pth",
            trigger=MaxValueTrigger(mode="validation", key="total acc")),
        HistorySaver(
            directory=p,
            name=lambda x: "history.pth",
            trigger=IntervalTrigger(period=1)),
        HistoryLogger(trigger=IntervalTrigger(period=1), print_func=print),
        LearningCurvePlotter(
            directory=p,
            trigger=IntervalTrigger(period=1),
        )
    ]

    # 学習
    trainer = Trainer(
        net,
        optimizer,
        criterion,
        train_loader,
        scheduler=scheduler,
        extensions=extensions,
        evaluators=evaluators,
        device=device
    )
    trainer.train(epochs, val_loader)

    # モデル保存
    output_dir = Path("../model")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    torch.save(net.state_dict(), output_dir / "final_model.pth")


if __name__ == "__main__":
    main()
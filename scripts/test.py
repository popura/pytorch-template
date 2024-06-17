from pathlib import Path
import os
import shutil
import argparse

import torch
import torchvision
from torchvision.transforms import v2 

from omegaconf import OmegaConf

from src.data_pipeline import DataPipeline
from src.model import SimpleCNN
from src.trainer import AccuracyEvaluator
from src.train_id import print_config
from src.util import set_random_seed


def main(
        cfg,
        train_id,
        seed
    ):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_config(cfg)

    net = SimpleCNN(**cfg.model.params)
    net.load_state_dict(torch.load(f"/{os.environ['PROJECT_NAME']}/outputs/train/history/{train_id}/best_model.pth"))
    net = net.to(device)

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])
    dataset = torchvision.datasets.MNIST(
        root="./data/",
        train=False,
        download=True,
    )
    classes = dataset.classes
    datapipe = DataPipeline(dataset, static_transforms=transforms, dynamic_transforms=None, max_cache_size=0)
    test_loader = torch.utils.data.DataLoader(
        datapipe,
        shuffle=False,
        batch_size=64,
        num_workers=0
    )

    net.eval()
    evaluator = AccuracyEvaluator(classes)
    evaluator.initialize()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        evaluator.eval_batch(outputs, targets)
    result = evaluator.finalize()
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default=f"/{os.environ['PROJECT_NAME']}/outputs/train/history",
                        help='Directory path for searching trained models')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Sets random seed')
    parser.add_argument('--dataset_dir', type=str,
                        default=None,
                        help='Sets random seed')
    parser.add_argument('--remove_untrained_id', action="store_true",
                        help='If True, history directories in history_dir'
                        +'that do not contain model checkpoints will be removed')
    parser.add_argument('--skip_tested', action="store_true",
                        help='If True, train IDs whose resulting directories already exist'
                        +'will be skipped')

    args = parser.parse_args()
    p = Path.cwd() / args.history_dir
    for q in sorted(p.glob('**/config.yaml')):
        cfg = OmegaConf.load(str(q))
        train_id = q.parent.name

        if not (q.parent / "best_model.pth").exists():
            print(f"A model for Train ID {train_id} does not exist")
            if args.remove_untrained_id:
                shutil.rmtree(q.parent)
            continue

        result_dir = Path(f"/{os.environ['PROJECT_NAME']}") / "outputs" / Path(__file__).stem
        result_dir /= train_id
        if result_dir.exists():
            print(f"Train ID {train_id} is already tested")
            if args.skip_tested:
                continue
        else:
            result_dir.mkdir(parents=True, exist_ok=True)

        try:
            main(cfg=cfg,
                 train_id=train_id,
                 seed=args.seed)
        except Exception as e:
            print(f"Train ID {train_id} is skipped due to an exception {e}")

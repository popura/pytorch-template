from abc import ABCMeta, abstractmethod
from typing import Any
from multiprocessing import Manager

import torch


class NumpiedTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.array = tensor.numpy()

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.array)


def numpize_sample(sample: Any) -> Any:
    if isinstance(sample, torch.Tensor):
        return NumpiedTensor(sample)
    elif isinstance(sample, tuple):
        return tuple(numpize_sample(s) for s in sample)
    elif isinstance(sample, list):
        return [numpize_sample(s) for s in sample]
    elif isinstance(sample, dict):
        return {k: numpize_sample(v) for k, v in sample.items()}
    else:
        return sample


def tensorize_sample(sample: Any) -> Any:
    if isinstance(sample, NumpiedTensor):
        return sample.to_tensor()
    elif isinstance(sample, tuple):
        return tuple(tensorize_sample(s) for s in sample)
    elif isinstance(sample, list):
        return [tensorize_sample(s) for s in sample]
    elif isinstance(sample, dict):
        return {k: tensorize_sample(v) for k, v in sample.items()}
    else:
        return sample


class DataPipeline(torch.utils.data.Dataset):
    def __init__(self,
                 ds: torch.utils.data.Dataset,
                 static_transforms=None,
                 dynamic_transforms=None,
                 max_cache_size=None):
        super().__init__()
        self.ds = ds
        self.static_transforms = static_transforms
        self.dynamic_transforms = dynamic_transforms

        self.cache = Manager().dict()
        self.cache_size = 0
        self.max_cache_size = max_cache_size if max_cache_size is not None else len(ds)

    def __getitem__(self, idx):
        if idx in self.cache:
            x = self.cache[idx] # 共有テンソルをコピー
            x = tensorize_sample(x)
        else:
            x = self.ds[idx]
            if self.static_transforms is not None:
                x = self.static_transforms(*x)
            if self.cache_size + 1 < self.max_cache_size:
                self.cache[idx] = numpize_sample(x)
                self.cache_size += 1
        if self.dynamic_transforms is not None:
            x = self.dynamic_transforms(*x)
        return x

    def __len__(self):
        return len(self.ds)


class Transform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x):
        pass


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Zip(Transform):
    def __init__(self, *transforms):
        self.transforms = transforms
    
    def __call__(self, *args):
        outputs = []
        for i in range(len(args)):
            outputs.append(self.transforms[i](args[i]))
        return tuple(outputs)


class Identity(Transform):
    def __call__(self, sample):
        return sample


from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch


class DataPipeline(torch.utils.data.Dataset):
    def __init__(self,
                 ds: torch.utils.data.Dataset,
                 static_transforms=None,
                 dynamic_transforms=None):
        self.ds = ds
        self.static_transforms = static_transforms
        self.dynamic_transforms = dynamic_transforms

    def __getitem__(self, index):
        raise NotImplementedError()
    
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


from abc import ABCMeta, abstractmethod
import torch


class Transform(metaclass = ABCMeta):
    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError()


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

    def __call__(self, waveform):
        for t in self.transforms:
            waveform = t(waveform)
        return waveform

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
    def __call__(self, x):
        return x

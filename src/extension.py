from pathlib import Path
import typing
import pickle
import time

import torch
import torch.nn as nn

from src.trainer import Trainer


class Trigger(object):
    """A Base class of triggers used for calling extension modules in training.

    Args:
    """
    def __init__(self) -> typing.NoReturn:
        super().__init__()
    
    def __call__(self, trainer: Trainer) -> bool:
        return True


class IntervalTrigger(Trigger):
    """Trigger based on a fixed interval.
    This trigger accepts iterations divided by a given interval.

    Args:
        period: Length of the interval
    """

    def __init__(self, period: int) -> typing.NoReturn:
        super().__init__()
        self.period = period
        self._previous_epoch = 0.

    def __call__(self, trainer: Trainer) -> bool:
        """Decides whether the extension should be called on this iteration.
        Args:
            trainer (trainer.Trainer):
                Trainer object that this trigger is associated with.
                The epoch information in this trainer is used to
                determine if the trigger should fire.
        Returns:
            True if the corresponding extension should be invoked in this
            iteration.
        """
        epoch = trainer.epoch
        previous_epoch = self._previous_epoch
        fire = previous_epoch // self.period != epoch // self.period
        self._previous_epoch = trainer.epoch

        return fire


class BestValueTrigger(Trigger):
    """Trigger based on the best value.

    Args:
        mode: ``train`` or ``validation``.
        key: Key of compared value.
        compare: Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value.
    """

    def __init__(self, mode:str, key:str, compare:typing.Callable[[], bool],
                 trigger: IntervalTrigger=None) -> typing.NoReturn:
        super().__init__()
        self.mode = mode
        self.key = key
        self.compare = compare
        self.best_value = None
        if trigger is None:
            trigger = IntervalTrigger(1)
        self.trigger = trigger
    
    def __call__(self, trainer: Trainer) -> bool:
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer: Trainer object that this trigger is associated with.
                The ``history`` of this trainer
                is used to determine if the trigger should fire.

        Returns:
            ``True`` if the corresponding extension should be invoked in
            this iteration.
        """

        if not self.trigger(trainer):
            return False

        history = trainer.history[self.mode][-1]
        key = self.key
        value = float(history[key])  # copy to CPU

        if self.best_value is None or self.compare(self.best_value, value):
            self.best_value = value
            return True
        return False


class MaxValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes maximum.
    For example you can use this trigger to take snapshot on the epoch the
    validation accuracy is maximum.

    Args:
        mode: ``train`` or ``validation``.
        key: Key of compared value.
        trigger: Trigger that decides the comparison interval between current
            best value and new value.
    """

    def __init__(self, mode:str, key:str, trigger: IntervalTrigger=None) -> typing.NoReturn:
        super().__init__(
            mode, key,
            lambda max_value, new_value: new_value > max_value, trigger)


class MinValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes minimum.
    For example you can use this trigger to take snapshot on the epoch the
    validation loss is minimum.

    Args:
        mode: ``train`` or ``validation``.
        key: Key of compared value.
        trigger: Trigger that decides the comparison interval between current
            best value and new value.
    """

    def __init__(self, mode:str, key:str, trigger: IntervalTrigger=None) -> typing.NoReturn:
        super().__init__(
            mode, key,
            lambda max_value, new_value: new_value < max_value, trigger)


class Extension(object):
    """A Base class of extensions for training models.

    Args:
        trigger: 
    """
    def __init__(self, trigger: Trigger) -> typing.NoReturn:
        super().__init__()
        self.trigger = trigger

    def initialize(self, trainer: Trainer) -> typing.NoReturn:
        """ Automatically called before training. Optional.

        Args:

        Returns:
        
        """
        pass

    def finalize(self, trainer: Trainer) -> typing.NoReturn:
        """ Automatically called after training. Optional.

        Args:

        Returns:
        """
        pass

    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        """ Called when the associated trigger is fired.
        """
        raise NotImplementedError()

    def state_dict(self):
        """ Used to serialize the state. Optional.
        """
        pass

    def load_state_dict(self, state):
        """ Used to deserialize the state. Optional.
        """
        pass


class ModelSaver(Extension):
    """Extension saving intermediate models in training.

    Args:
        directory: A directory where models will be saved.
        name: A function that returns a file name for saved model.
        trigger:
    """

    def __init__(self, directory: Path,
                 name: typing.Callable[[Trainer], str],
                 trigger: Trigger) -> typing.NoReturn:
        super().__init__(trigger)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.name = name
        self.trigger = trigger
    
    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        net = trainer.net
        if isinstance(net, nn.DataParallel):
            net = net.module
        name = self.name(trainer)
        torch.save(net.state_dict(), str(self.directory / name))


class HistorySaver(Extension):
    """Extension saving history of training.

    Args:
        directory: A directory where history will be saved.
        name: A function that returns a file name for saved history.
        trigger:
    """

    def __init__(self, directory: Path,
                 name: typing.Callable[[Trainer], str],
                 trigger: Trigger) -> typing.NoReturn:
        super().__init__(trigger)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.name = name
        self.trigger = trigger
    
    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        name = self.name(trainer)
        history = trainer.history

        with open(str(self.directory / name), mode="wb") as f:
            pickle.dump(history, f)


class HistoryLogger(Extension):
    """Extension logging history of training.

    Args:
        trigger:
    """

    def __init__(self, trigger: Trigger, print_func = None) -> typing.NoReturn:
        super().__init__(trigger)
        self.trigger = trigger
        self._start_time = None
        self.print_func = print_func or print
    
    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        history = trainer.history
        epoch = trainer.epoch
        
        elapsed_time = time.time() - self._start_time
        ave_required_time = elapsed_time / epoch
        finish_time = ave_required_time * (trainer.total_epoch - epoch)
        format_str = f"epoch: {epoch:03d}/{trainer.total_epoch:03d}"
        format_str += " | "
        format_str += f"loss: {history['train'][-1]['loss']:.4f}"
        format_str += " | "
        if history["validation"] is not None:
            for k, v in history["validation"][-1].items():
                if k == "epoch":
                    continue
                format_str += f"val. {k}: {v:.4f}"
                format_str += " | "
        format_str += f"time: {int(elapsed_time/60/60):02d} hour {elapsed_time/60%60:02.2f} min"
        format_str += " | "
        format_str += f"finish after: {int(finish_time/60/60):02d} hour {finish_time/60%60:02.2f} min"
        self.print_func(format_str)
    
    def initialize(self, trainer: Trainer) -> typing.NoReturn:
        self._start_time = time.time()
    
    def finalize(self, trainer: Trainer) -> typing.NoReturn:
        elapsed_time = time.time() - self._start_time
        self.print_func(f"Total training time: {int(elapsed_time/60/60):02d} hour {elapsed_time/60%60:02.2f} min")


class LearningCurvePlotter(Extension):
    """Extension plotting learning curves.

    Args:
        directory: A directory where learning curves will be saved.
        trigger:
    """

    def __init__(self, directory: Path, trigger: Trigger) -> typing.NoReturn:
        super().__init__(trigger)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.trigger = trigger
    
    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        import matplotlib.pyplot as plt
        history = trainer.history
        train_losses = [h["loss"] for h in history["train"]]
        val_losses = [h["loss"] for h in history["validation"]]
        epoch = [h["epoch"] for h in history["train"]]

        plt.plot(epoch, train_losses, label="train")
        plt.plot(epoch, val_losses, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(str(self.directory / "learning_curve.png"))
        plt.clf()
        plt.close()

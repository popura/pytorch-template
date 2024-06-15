from pathlib import Path
import torch
import torchaudio



class AcousticSceneDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transforms=None):
        """
        Args:
            root: path to the dataset directory
                  ("pytorch-introduction/data/small-acoustic-scenes")
            mode: "train", "validation", or "test"
            transforms: an instance of Transform class (see transform.py),
                        which is used for preprocessing data
        Returns:
            None
        """
        self.root = Path(root)
        if mode in ("train", "validation", "test"):
            self.mode = mode
        else:
            raise ValueError()
        self.LABELS = ("car", "home")
        self.label2int = {label: i for i, label in enumerate(self.LABELS)}
        self.transforms = transforms
        
        # 入力データのPathとその出力データ（のPath）からなるタプルのリストを作る
        self.samples = self.make_dataset()
    
    def make_dataset(self, ):
        """
        Args:
            self: 
        Returns:
            samples: a list of tuples containing a path to an input signal and its label (as an integer)
        """
        samples = []
        for label in self.LABELS:
            for p in sorted((self.root / self.mode / label).glob("*.wav")):
                samples.append((p, self.label2int[label]))
        
        return samples
    
    def __getitem__(self, index: int):
        """引数として番号（インデックス）を受け取って
        その番号に対応したデータを読み込む
        Args: 
            index: an integer where 0 <= index < len(self)
        Returns:
            waveform: an instance of torch.Tensor that will be fed into a DNN after batched
            target: an integer representing a label
        """
        path, target = self.samples[index]
        waveform, sampling_rate = torchaudio.load(path)
        if self.transforms is not None:
            waveform, target = self.transforms(waveform, target)
        return waveform, target
    
    def __len__(self, ):
        """データセット全体のデータ数を返す
        Args:
            None
        Returns:
            the length (i.e., the number of data samples) of this dataset
        """
        return len(self.samples)

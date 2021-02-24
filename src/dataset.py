import sys
import os
import os.path
import re
import random
from typing import List, Dict
import pickle
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import ContentTooShortError
import tarfile
import zipfile
import shutil

import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset

from deepy.data.dataset import PureDatasetFolder, has_file_allowed_extension
from deepy.data.transform import SeparatedTransform
from deepy.data.audio.audiodataset import AUDIO_EXTENSIONS, default_loader


class DCASE2018Task5Dataset(PureDatasetFolder):
    """ DCASE2018 challenge task5 dataset for event recognition from multi-channel audio.
        This dataset is based on the SINS dataset.
    """
    def __init__(self, root, mode, node=None,
                 loader=default_loader, extensions=AUDIO_EXTENSIONS,
                 transforms=None, transform=None, target_transform=None,
                 is_valid_file=None,
                 pre_load=False, pre_transform=None,
                 pre_target_transform=None, pre_transforms=None):
        super(DCASE2018Task5Dataset, self).__init__(root,
                                                    transforms=transforms,
                                                    transform=transform,
                                                    target_transform=target_transform)
        self.MODES = ('train', 'evaluate', 'test')

        if mode not in self.MODES:
            raise ValueError("mode \"{}\" is not in {}".format(mode, self.MODES))
        self.mode = mode

        if node is None:
            node = list(range(2, 5))
        self.node = node

        meta_dir = (Path(root) / "DCASE2018-task5-dev" / "meta")
        if not meta_dir.exists():
            meta_dir.mkdir(parents=True, exist_ok=True)
            self._modify_meta_info(directory=str(self.root))

        classes, class_to_idx = self._define_classes()
        samples = self._make_dataset(str(self.root), mode, node,
                                     class_to_idx, 1, extensions, is_valid_file)
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx

        has_pre_transforms = pre_transforms is not None
        has_pre_separate_transform = pre_transform is not None or pre_target_transform is not None
        if has_pre_transforms and has_pre_separate_transform:
            raise ValueError("Only pre_transforms or pre_transform/pre_target_transform can "
                             "be passed as argument")
        if has_pre_separate_transform:
            pre_transforms = SeparatedTransform(pre_transform, pre_target_transform)
        self.pre_transforms = pre_transforms
        self.pre_load = pre_load
        if pre_load:
            self.pre_process()

    def pre_process(self, ):
        preprocessed_samples = []
        for i in range(len(self)):
            sys.stdout.write("\rloaded {0} / {1}".format(i+1, len(self)))
            sys.stdout.flush()
            path, target = self.samples[i]
            sample = self.loader(path)
            if self.pre_transforms is not None:
                sample, target = self.pre_transforms(sample, target)
            preprocessed_samples.append((sample, target))

        self.preprocessed_samples = preprocessed_samples
        sys.stdout.write("\n")
    
    def _define_classes(self, ):
        classes = ["absence",
                   "cooking",
                   "dishwashing",
                   "eating",
                   "other",
                   "social_activity",
                   "vacuum_cleaner",
                   "watching_tv",
                   "working"]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def _make_dataset(self, directory: str, mode: str, node: List[int], class_to_idx: Dict[str, int],
                      fold: int=1, extensions=None, is_valid_file=None):
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))

        sub_directory = os.path.join(directory, 'DCASE2018-task5-dev')
        meta_directory = os.path.join(sub_directory, 'meta')

        df = pd.read_csv(os.path.join(meta_directory, "fold{}_{}_meta.csv".format(fold, mode)), index_col=0)
        columns = ['node{}'.format(i) for i in node]
        columns.append("class")
        df = df[columns]
        instances = []
        for row in df.itertuples():
            for i in range(len(node)):
                path = os.path.join(sub_directory, str(row[i+1]))
                if is_valid_file(path):
                    print((path, row[-1]))
                    class_index = class_to_idx[row[-1]]
                    instances.append((path, class_index))

        return instances

    def _parse_meta_info(self, directory:str, mode:str, fold: int=1):
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))

        sub_directory = os.path.join(directory, 'DCASE2018-task5-dev')
        with open(os.path.join(sub_directory, 'evaluation_setup', 'fold{}_{}.txt'.format(fold, mode))) as f:
            lines = f.readlines()
        
        folders = []
        fnames = []
        classes = []
        nodes = []
        sessions = []
        segments = []
        with open(os.path.join(sub_directory, 'meta.txt')) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.rstrip('\n')
                path, class_name, session = line.split('\t')
                folder, fname = path.split('/')
                node_id, session_id, segment_id = re.findall('\d+', fname)
                
                if [tmpline for tmpline in lines if path in tmpline]:
                    folders.append(folder)
                    fnames.append(fname)
                    classes.append(class_name)
                    nodes.append(int(node_id))
                    sessions.append(int(session_id))
                    segments.append(int(segment_id))
        
        return folders, fnames, classes, nodes, sessions, segments

    def _modify_meta_info(self, directory: str):
        instances = []
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))

        for mode in ["train", "evaluate", "test"]:
            for fold in range(1, 5):
                folders, fnames, classes, nodes, sessions, segments = self._parse_meta_info(directory, mode=mode, fold=fold)

                path_dict = {'node{}'.format(i): {} for i in range(1, 5)}
                class_dict = {}
                for i in range(len(fnames)):
                    if (sessions[i], segments[i]) in class_dict:
                        if class_dict[(sessions[i], segments[i])] != classes[i]:
                            raise ValueError("Some classes are assigned for session {} segment {}".format(sessions[i], segments[i]))
                    else:
                        class_dict[(sessions[i], segments[i])] = classes[i]
            
                    path = os.path.join(folders[i], fnames[i])
                    path_dict["node{}".format(nodes[i])][(sessions[i], segments[i])] = path
        
                meta = {'node{}'.format(i): [] for i in range(1, 5)}
                class_list = []
                segment_list = sorted(set(zip(sessions, segments)))
                for session, segment in segment_list:
                    if (session, segment) in class_dict:
                        class_list.append(class_dict[(session, segment)])
                    else:
                        raise ValueError("Class for session {} segment {} is undefined".format(session, segment))
                    for k in path_dict.keys():
                        if (session, segment) in path_dict[k]:
                            meta[k].append(path_dict[k][(session, segment)])
                        else:
                            meta[k].append("")
                meta["class"] = class_list
                df = pd.DataFrame(data=meta, index=segment_list)
                df.to_csv(os.path.join(directory, "DCASE2018-task5-dev", "meta", "fold{}_{}_meta.csv".format(fold, mode)))
        return 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.pre_load:
            sample, target = self.preprocessed_samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)


def download_file(remote_file, local_file, exist_ok=False):
    """
    Download single file to local_dir
    Args:
        remote_file:
        local_file:
        exist_ok:
    Returns:
    """
    local_file = Path(local_file)
    if not local_file.exists():
        def progress_hook(t):
            """
            https://raw.githubusercontent.com/tqdm/tqdm/master/examples/tqdm_wget.py
            
            Wraps tqdm instance. Don't forget to close() or __exit__()
            the tqdm instance once you're done with it (easiest using
            `with` syntax).
            """

            last_b = 0

            def inner(b=1, bsize=1, tsize=None):
                """
                b  : int, optional
                    Number of blocks just transferred [default: 1].
                bsize  : int, optional
                    Size of each block (in tqdm units) [default: 1].
                tsize  : int, optional
                    Total size (in tqdm units). If [default: None]
                    remains unchanged.
                """
                nonlocal last_b
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b) * bsize)
                last_b = b

            return inner

        tmp_file = str(local_file) + '.tmp'
        with tqdm(
                desc="{0: >25s}".format(Path(remote_file).stem),
                file=sys.stdout,
                unit='B',
                unit_scale=True,
                miniters=1,
                leave=False,
                ascii=True
        ) as t:
            urlretrieve(
                str(remote_file),
                filename=tmp_file,
                reporthook=progress_hook(t),
                data=None
            )
        os.rename(tmp_file, local_file)
    elif not exist_ok:
        raise FileExistsError(local_file)
    return local_file


def extract_file(local_file, exist_ok=False):
    """
    If local_file is .zip or .tar.gz files are extracted.
    Args:
        local_file:
        exist_ok:
    Returns:
    """
    local_file = Path(local_file)
    local_dir = local_file.parent
    if local_file.exists():

        if local_file.name.endswith('.zip'):
            with zipfile.ZipFile(local_file, "r") as z:
                # Start extraction
                members = z.infolist()
                for i, member in enumerate(members):
                    target_file = local_dir / member.filename
                    if not target_file.exists():
                        try:
                            z.extract(member=member, path=local_dir)
                        except KeyboardInterrupt:
                            # Delete latest file, since most likely it
                            # was not extracted fully
                            if target_file.exists():
                                os.remove(target_file)
                            raise
                    elif not exist_ok:
                        raise FileExistsError(target_file)

        elif local_file.name.endswith('.tar.gz'):
            with tarfile.open(local_file, "r:gz") as tar:
                for i, tar_info in enumerate(tar):
                    target_file = local_dir / tar_info.name
                    if not target_file.exists():
                        try:
                            tar.extract(tar_info, local_dir)
                        except KeyboardInterrupt:
                            # Delete latest file, since most likely it
                            # was not extracted fully
                            if target_file.exists():
                                os.remove(target_file)
                            raise
                    elif not exist_ok:
                        raise FileExistsError(target_file)
                    tar.members = []



def download_file_list(file_list, target_dir, exist_ok=False, logger=None):
    """
    Download file_list to target_dir
    Args:
        file_list:
        target_dir:
        exist_ok:
        logger:
    Returns:
    """

    target_dir = Path(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    item_progress = tqdm(
        file_list, desc="{0: <25s}".format('Download files'),
        file=sys.stdout, leave=False, ascii=True)

    local_files = list()
    for remote_file in item_progress:
        try:
            local_files.append(
                download_file(
                    remote_file,
                    target_dir / Path(remote_file).name,
                    exist_ok=exist_ok
                )
            )
        except FileExistsError:
            local_files.append(target_dir / Path(remote_file).name)
        except ContentTooShortError:
            os.remove(target_dir / (str(Path(remote_file).name)) + '.tmp')
            local_files.append(
                download_file(
                    remote_file,
                    target_dir / Path(remote_file).name,
                    exist_ok=exist_ok
                )
            )
            

    item_progress = tqdm(
        local_files,
        desc="{0: <25s}".format('Extract files'),
        file=sys.stdout,
        leave=False,
        ascii=True
    )

    if logger is not None:
        logger.info('Starting Extraction')
    for _id, local_file in enumerate(item_progress):
        if local_file and local_file.exists():
            if logger is not None:
                logger.info(
                    '  {title:<15s} [{item_id:d}/{total:d}] {package:<30s}'
                    .format(
                        title='Extract files ',
                        item_id=_id,
                        total=len(item_progress),
                        package=local_file
                    )
                )
            extract_file(local_file, exist_ok=exist_ok)


class SINSDataset(PureDatasetFolder):
    """ the SINS dataset.
    """
    def __init__(self, root: str, mode: str, node: List[int]=None,
                 loader=default_loader, extensions=AUDIO_EXTENSIONS,
                 transforms=None, transform=None, target_transform=None,
                 is_valid_file=None,
                 pre_load=False, pre_transform=None,
                 pre_target_transform=None, pre_transforms=None,
                 download=False):
        super(SINSDataset, self).__init__(root,
                                          transforms=transforms,
                                          transform=transform,
                                          target_transform=target_transform)
        self.MODES = ('train', 'evaluate', 'test')
        self.RECORDS = {
            1: '2546677',
            2: '2547307',
            3: '2547309',
            4: '2555084',
            6: '2547313',
            7: '2547315',
            8: '2547319',
            9: '2555080',
            10: '2555137',
            11: '2558362',
            12: '2555141',
            13: '2555143'
        }

        if mode not in self.MODES:
            raise ValueError("mode \"{}\" is not in {}".format(mode, self.MODES))
        self.mode = mode

        if node is None:
            node = list(range(1, 14))
        self.node = node

        if download:
            self.download(self.node, self.root)
        #classes, class_to_idx = self._define_classes(str(self.root))
        #samples = self._make_dataset(str(self.root), mode, node,
        #                             class_to_idx, extensions, is_valid_file)
        samples = []
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx

        has_pre_transforms = pre_transforms is not None
        has_pre_separate_transform = pre_transform is not None or pre_target_transform is not None
        if has_pre_transforms and has_pre_separate_transform:
            raise ValueError("Only pre_transforms or pre_transform/pre_target_transform can "
                             "be passed as argument")
        if has_pre_separate_transform:
            pre_transforms = SeparatedTransform(pre_transform, pre_target_transform)
        self.pre_transforms = pre_transforms
        self.pre_load = pre_load
        if pre_load:
            self.pre_process()

    def pre_process(self, ):
        preprocessed_samples = []
        for i in range(len(self)):
            sys.stdout.write("\rloaded {0} / {1}".format(i+1, len(self)))
            sys.stdout.flush()
            path, target = self.samples[i]
            sample = self.loader(path)
            if self.pre_transforms is not None:
                sample, target = self.pre_transforms(sample, target)
            preprocessed_samples.append((sample, target))

        self.preprocessed_samples = preprocessed_samples
        sys.stdout.write("\n")
    
    def _define_classes(self, directory):
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))
        _, _, train_classes, *_ = self._parse_meta_info(directory, mode="train")
        _, _, test_classes, *_ = self._parse_meta_info(directory, mode="evaluate")
        classes = list(set(train_classes) | set(test_classes))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def _make_dataset(self, directory: str, mode: str, node: List[int], class_to_idx: Dict[str, int],
                      extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))

        folders, fnames, classes, nodes, sessions, segments = self._parse_meta_info(directory, mode=mode)
        loading_indices = [i for i in range(len(nodes)) if nodes[i] in node]

        path_dict = {}
        for i in loading_indices:
            path = os.path.join(folders[i], fnames[i])
            class_index = class_to_idx[classes[i]]
            path_dict[(sessions[i], segments[i])] = path, class_index
        
        for k, v in sorted(path_dict.items()):
            item = v[0], v[1]
            instances.append(item)

        return instances

    def _parse_meta_info(self, directory, mode):
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))

        sub_directory = os.path.join(directory, 'DCASE2018-task5-dev')
        if mode == "train":
            with open(os.path.join(sub_directory, 'evaluation_setup', 'fold1_train.txt')) as f:
                lines = f.readlines()
        elif mode == "evaluate":
            with open(os.path.join(sub_directory, 'evaluation_setup', 'fold1_evaluate.txt')) as f:
                lines = f.readlines()
        elif mode == "test":
            with open(os.path.join(sub_directory, 'evaluation_setup', 'fold1_test.txt')) as f:
                lines = f.readlines()
        else:
            raise NotImplementedError()
        
        folders = []
        fnames = []
        classes = []
        nodes = []
        sessions = []
        segments = []
        with open(os.path.join(sub_directory, 'meta.txt')) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.rstrip('\n')
                path, class_name, session = line.split('\t')
                folder, fname = path.split('/')
                node_id, session_id, segment_id = re.findall('\d+', fname)
                
                if [tmpline for tmpline in lines if path in tmpline]:
                    folders.append(os.path.join(sub_directory, folder))
                    fnames.append(fname)
                    classes.append(class_name)
                    nodes.append(int(node_id))
                    sessions.append(int(session_id))
                    segments.append(int(segment_id))
        
        return folders, fnames, classes, nodes, sessions, segments
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.pre_load:
            sample, target = self.preprocessed_samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)

    def download(self, nodes: List[int], destination: str):
        """
        Download dataset over the internet to the local path
        Args:
            nodes: list of nodes to download
            destination: local path to store dataset at.
        Returns:
        """
        destination = Path(destination)
        os.makedirs(destination, exist_ok=True)

        # Download git repository with annotations and matlab code
        try:
            extract_file(download_file(
                'https://github.com/KULeuvenADVISE/SINS_database/archive/master.zip',
                destination / 'SINS_database-master.zip'
            ))
        except FileExistsError:
            pass

        # Download node data from zenodo repositories
        item_progress = tqdm(
            nodes, desc="{0: <25s}".format('Download nodes'),
            file=sys.stdout, leave=False, ascii=True)

        for node_id in item_progress:
            self.download_node(node_id, destination)
        shutil.move(
            str(destination / 'Original' / 'audio'),
            str(destination / 'audio')
        )
        os.rmdir(str(destination / 'Original'))
        for file in (destination / 'SINS_database-master').iterdir():
            shutil.move(str(file), str(destination))
        os.rmdir(str(destination / 'SINS_database-master'))
        
    def download_node(self, node_id: int, destination: str):
        """
        Download data from single node to destination
        Args:
            node_id:
            destination:
        Returns:
        """

        files = [
            f'https://zenodo.org/record/{self.RECORDS[node_id]}/files/{file}'
            for file in [
                'Node{}_audio_{:02}.zip'.format(node_id, j) if node_id < 10
                else 'Node{}_audio_{}.zip'.format(node_id, j)
                for j in range(1, 10 + (node_id < 9))
            ] + ['license.pdf'] + (['readme.txt'] if node_id != 1 else [])
        ]
        download_file_list(files, destination, exist_ok=True)

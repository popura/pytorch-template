import argparse
import hashlib
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf, DictConfig



def print_config(cfg: DictConfig) -> None:
    print('-----Parameters-----')
    print(OmegaConf.to_yaml(cfg))
    print('--------------------')
    return


def is_same_config(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    """Compare cfg1 with cfg2.

    Args:
        cfg1: Config
        cfg2: Config

    Returns:
        True if cfg1 == cfg2 else False

    """
    return cfg1 == cfg2


def generate_train_id(cfg: DictConfig) -> str:
    """Generate unique ID for a training condition specified by cfg.

    Args:
        cfg: Config for training

    Returns:
        train_id: Unique ID

    """
    train_id = hashlib.md5(str(cfg).encode()).hexdigest()[:10]
    return train_id


def enumerate_dictkeys(d: DictConfig):
    keys = []
    for k, v in sorted(d.items()):
        if isinstance(v, DictConfig):
            tmp_keys = [k + '.' + key for key in enumerate_dictkeys(v)]
            keys.extend(tmp_keys)
        else:
            keys.append(k)
    return keys


def list_train_id(cfgs, keys):
    df = pd.DataFrame(columns=['Train ID'].extend(keys))
    if isinstance(cfgs, list):
        for train_id, cfg in cfgs:
            d = {'Train ID': train_id}
            d.update({key: get_cfg_value(cfg, key) for key in keys})
            df = df.append(d, ignore_index=True)
    else:
        train_id, cfg = cfgs
        d = {'Train ID': train_id}
        d.update({key: get_cfg_value(cfg, key) for key in keys})
        df = df.append(d, ignore_index=True)
    
    return df


def get_cfg_value(d: DictConfig, key: str):
    tmpd = d
    for k in key.split('.'): 
        tmpd = tmpd[k]
    
    return tmpd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default='./history',
                        help='Directory path for searching trained models')
    parser.add_argument('--keys', default=None, nargs='+',
                        help='Keys of DictConfig for filtering')              
    args = parser.parse_args()
    p = Path.cwd() / args.history_dir

    cfg_list = []
    if args.keys is None:
        keys = []
    else:
        keys = args.keys

    for q in p.glob('**/config.yaml'):
        cfg = OmegaConf.load(str(q))
        train_id = q.parent.name
        cfg_list.append((train_id, cfg))

        if args.keys is None:
            keys.extend(enumerate_dictkeys(cfg))
    
    keys = list(set(keys))
    df = list_train_id(cfg_list, keys)
    df.to_csv(str(Path.cwd() / 'train_id.csv'))
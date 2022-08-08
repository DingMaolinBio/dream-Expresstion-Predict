import logging
logger = logging.getLogger(__name__)

from typing import Any, Iterable, Tuple, List, Union
import sklearn.metrics as metrics  # auc, precision_recall_curve, roc_auc_score
import numpy as np
import warnings
import torch
from torch import Tensor


def roc_auc_score(y_true, probas_pred, robust: bool = False) -> float:
    if robust and len(np.unique(y_true)) == 1:
        res = 0.5
    else:
        res = metrics.roc_auc_score(y_true, probas_pred)
    return res


def pr_auc_score(y_true, probas_pred, robust: bool = False):
    if robust and len(np.unique(y_true)) == 1:
        res = 0
    else:
        p, r, _ = metrics.precision_recall_curve(y_true, probas_pred)
        res = metrics.auc(r, p)
    return res


def _bce_numpy(y_true: np.ndarray, probas_pred: np.ndarray, eps=1E-15) -> float:
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape,
                          probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape,
                                                                                                           probas_pred.shape)
    dtype = (1.0 + y_true[0]).dtype  # at least np.float16
    y_true = y_true.astype(dtype)
    probas_pred = probas_pred.astype(dtype)
    probas_pred = np.clip(probas_pred, a_min=eps, a_max=1 - eps)
    bce = -(y_true * np.log(probas_pred) + (1 - y_true) * np.log(1 - probas_pred))
    return np.mean(bce)


def _bce_torch(y_true: Tensor, probas_pred: Tensor, mask: Tensor = None, eps=1E-15) -> Tensor:
    assert np.array_equal(y_true.shape,
                          probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape,
                                                                                                           probas_pred.shape)
    dtype = (1.0 + y_true[0]).dtype  # at least np.float16
    y_true = torch.as_tensor(y_true, dtype=dtype)
    probas_pred = torch.as_tensor(probas_pred, dtype=dtype).clamp(min=eps, max=1 - eps)
    bce = -(y_true * torch.log(probas_pred) + (1 - y_true) * torch.log(1 - probas_pred))
    if mask is not None:
        bce *= mask
    return torch.mean(bce)


def binary_cross_entropy(y_true: Union[np.ndarray, Tensor], probas_pred: Union[np.ndarray, Tensor], eps: float = 1E-15,
                         mask: Tensor = None) -> Union[float, Tensor]:
    if isinstance(y_true, Tensor):
        return _bce_torch(y_true=y_true, probas_pred=probas_pred, eps=eps, mask=mask)
    else:
        return _bce_numpy(y_true=y_true, probas_pred=probas_pred, eps=eps)


def roc_auc_score_2d(y_true, probas_pred, robust: bool = True) -> np.ndarray:
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape,
                          probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape,
                                                                                                           probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected"
    auc_list = list()
    for i in range(y_true.shape[0]):
        auc_list.append(roc_auc_score(y_true[i], probas_pred[i], robust=robust))
    return np.asarray(auc_list)


def pr_auc_score_2d(y_true: np.ndarray, probas_pred: np.ndarray, robust: bool = True) -> np.ndarray:
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape,
                          probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape,
                                                                                                           probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected"
    auc_list = list()
    for i in range(y_true.shape[0]):
        auc_list.append(pr_auc_score(y_true[i], probas_pred[i], robust=robust))
    return np.asarray(auc_list)


def roc_pr_auc_scores_2d(y_true: np.ndarray, probas_pred: np.ndarray, robust: bool = True) -> Tuple[float, float]:
    """
    Return:
    auc(mean) : float
    auprc(mean) : float
    """
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape,
                          probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape,
                                                                                                           probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected, while encountered {}".format(y_true.shape)
    auc_list, auprc_list = list(), list()
    for i in range(y_true.shape[0]):
        auc_list.append(roc_auc_score(y_true[i], probas_pred[i], robust=robust))
        auprc_list.append(pr_auc_score(y_true[i], probas_pred[i], robust=True))
    return np.nanmean(auc_list), np.nanmean(auprc_list)


def subset_indices(groups: Iterable[int], subset: List[str], complement: bool = False) -> Iterable[int]:
    """
    Args
        groups : the group of samples
        subset:
    """
    if not complement:
        indices = np.arange(len(groups))[np.isin(groups, list(subset))]
    else:
        indices = np.arange(len(groups))[np.logical_not(np.isin(groups, list(subset)))]
    return indices


def _split_train_valid_test(groups, train_keys, valid_keys, test_keys=None):
    """
    groups: length N, the number of samples
    train
    """
    assert isinstance(train_keys, list)
    assert isinstance(valid_keys, list)
    assert test_keys is None or isinstance(test_keys, list)
    index = np.arange(len(groups))
    train_idx = index[np.isin(groups, train_keys)]
    valid_idx = index[np.isin(groups, valid_keys)]
    if test_keys is not None:
        test_idx = index[np.isin(groups, test_keys)]
        return train_idx, valid_idx, test_idx
    else:
        return train_idx, valid_idx


def split_train_valid_test(sample_number: int, val_ratio: float, test_ratio: float, stratify: Iterable = None) -> Tuple[
    Iterable[int], Iterable[int], Iterable[int]]:
    r"""
    Description
    ------------
    Randomly split train/validation/test data

    Arguments
    ---------
    stratify: split by groups

    Return
    -------
    train_inds : List[int]
    val_inds : List[int]
    test_inds : List[int]
    """
    val_ratio = 0 if val_ratio is None else val_ratio
    test_ratio = 0 if test_ratio is None else test_ratio
    assert val_ratio + test_ratio > 0 and val_ratio + test_ratio < 1, "{},{}".format(val_ratio, test_ratio)
    all_inds = np.arange(sample_number)
    from sklearn.model_selection import train_test_split

    train_val_inds, test_inds = train_test_split(all_inds, test_size=test_ratio, stratify=stratify)
    val_ratio = val_ratio / (1 - test_ratio)
    if stratify is not None:
        stratify = np.asarray(stratify)[train_val_inds]
    train_inds, val_inds = train_test_split(train_val_inds, test_size=val_ratio, stratify=stratify)

    return train_inds, val_inds, test_inds


def split_train_val_test_by_group(groups: List[Any], n_splits: int, val_folds: int, test_folds: int) -> Tuple[
    List, List, List]:
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=n_splits)
    train_inds, val_inds, test_inds = list(), list(), list()
    for i, (_, inds) in enumerate(splitter.split(groups, groups=groups)):
        if i < val_folds:
            train_inds.append(inds)
        elif i >= val_folds and i < test_folds:
            val_inds.append(inds)
        else:
            test_inds.append(inds)
    train_inds = np.concatenate(train_inds)
    if val_folds > 0:
        val_inds = np.concatenate(val_inds)
    if test_folds:
        test_inds = np.concatenate(test_inds)
    return train_inds, val_inds, test_inds


def idle_gpu(n: int=1, min_memory: int=4096, time_step: int=60, time_out: int=3600 * 16, skip: set=set()):
    import random, time, os
    from subprocess import Popen, PIPE
    if type(skip) is int:
        skip = {str(skip)}
    elaspsed_time = 0
    p = Popen(['/bin/bash', '-c', "nvidia-smi | grep GeForce | wc -l"], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    n_GPUs = int(out)
    random.seed(int(time.time()) % os.getpid())
    rand_priority = [random.random() for x in range(n_GPUs)]
    while elaspsed_time < time_out:
        cmd = "nvidia-smi | grep Default | awk '{print NR-1,$9,$11,$13,$3}' | sed 's/MiB//g;s/%//g;s/C//g'"
        p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        query_result, err = out.decode('utf8'), err.decode('utf8')
        rc = p.returncode
        query_result = query_result.strip().split('\n')

        gpu_list = list()
        for i, gpu_info in enumerate(query_result):
            gpu, memory_usage, memory_total, gpu_usage, temp = gpu_info.split(' ')
            if gpu in skip:
                continue
            memory_usage, memory_total, gpu_usage, temp = int(memory_usage), int(memory_total), int(gpu_usage), int(temp)
            memory = memory_total - memory_usage
            gpu_list.append((gpu, 500 * int(round(memory_usage / 500)) + rand_priority[i], int(round(gpu_usage/10)), int(round(temp / 10)), memory)) # reverse use
        ans = sorted(gpu_list, key=lambda x:(x[1], x[2], x[3]))
        if ans[0][-1] < min_memory:
            print("Waiting for available GPU... (%s)" % (time.asctime()))
            # time.sleep(60 * 10)
            time.sleep(time_step)
            elaspsed_time += time_step
            if elaspsed_time > time_out:
                raise MemoryError("Error: No available GPU with memory > %d MiB" % (min_memory))
        else:
            break

    #return ','.join(ans[0][0])
    return ','.join([ans[i][0] for i in range(n)])

def select_device(device_id=None):
    import os, torch
    ## device_id: None: auto / -1: cpu /
    if device_id is not None:
        try:
            d = int(device_id)
            device_id = d
        except ValueError:
            pass
    if device_id == -1 or str(device_id).lower() == "cpu":
        device_id = 'cpu'
        device = torch.device('cpu')
    else:
        if device_id is None:
            device_id = str(idle_gpu())
        else:
            device_id = str(device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device('cuda')
    return device_id, device

from io import TextIOWrapper
import gzip
def copen(fn: str, mode='rt') -> TextIOWrapper:
    if fn.endswith(".gz"):
        return gzip.open(fn, mode=mode)
    else:
        return open(fn, mode=mode)

NN_COMPLEMENT = {
    'A': 'T',
    'C': 'G',
    'G': 'C',
    'T': 'A',
    'R': 'Y',
    'Y': 'R',
    'S': 'S',
    'W': 'W',
    'K': 'M',
    'M': 'K',
    'B': 'V',
    'D': 'H',
    'H': 'D',
    'V': 'B',
    'N': 'N',
    '.': '.',
    '-': '-'
}

def get_reverse_strand(seq):
    seq = seq[::-1]
    seq = ''.join([NN_COMPLEMENT[n] for n in seq])
    return seq

_CHROM2INT = {
    "chr1": 1,   "1": 1,   1: 1,
    "chr2": 2,   "2": 2,   2: 2,
    "chr3": 3,   "3": 3,   3: 3,
    "chr4": 4,   "4": 4,   4: 4,
    "chr5": 5,   "5": 5,   5: 5,
    "chr6": 6,   "6": 6,   6: 6,
    "chr7": 7,   "7": 7,   7: 7,
    "chr8": 8,   "8": 8,   8: 8,
    "chr9": 9,   "9": 9,   9: 9,
    "chr10": 10, "10": 10, 10: 10,
    "chr11": 11, "11": 11, 11: 11,
    "chr12": 12, "12": 12, 12: 12,
    "chr13": 13, "13": 13, 13: 13,
    "chr14": 14, "14": 14, 14: 14,
    "chr15": 15, "15": 15, 15: 15,
    "chr16": 16, "16": 16, 16: 16,
    "chr17": 17, "17": 17, 17: 17,
    "chr18": 18, "18": 18, 18: 18,
    "chr19": 19, "19": 19, 19: 19,
    "chr20": 20, "20": 20, 20: 20,
    "chr21": 21, "21": 21, 21: 21,
    "chr22": 22, "22": 22, 22: 22,
    "chrX": 23,  "X": 23,
    "chrY": 24,  "Y": 24,
    "chrM": 25,  "M": 25
}
class Chrom2Int(object):
    def __init__(self) -> None:
        self.mapping = _CHROM2INT.copy()
        self.reverse_mapping = {v:k for k, v in self.mapping.items()}
        self._next = max(self.mapping.values()) + 1

    def __call__(self, chrom) -> int:
        if chrom not in self.mapping:
            self.mapping[chrom] = self._next
            self.reverse_mapping[self._next] = chrom
            self._next += 1
        return self.mapping[chrom]

from typing import Dict
class LabelEncoder(object):
    def __init__(self, predefined_mapping: Dict[str, int] = dict()) -> None:
        self.mapping = predefined_mapping.copy()
        if len(self.mapping) == 0:
            self._next = 0
        else:
            self._next = max(self.mapping.values()) + 1
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def __call__(self, label) -> int:
        if label not in self.mapping:
            self.mapping[label] = self._next
            self.reverse_mapping[self._next] = label
            self._next += 1
        return self.mapping[label]

    def id2label(self, id) -> str:
        return self.reverse_mapping[id]

import torch.nn as nn
def freeze_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module.eval()

import random
def set_rand_seed(seed=1, backends=True):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = backends
    torch.backends.cudnn.benchmark = backends
    torch.backends.cudnn.deterministic = not backends
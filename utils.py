import logging
import torch
import numpy as np


def gather_metrics(metrics):
    _metrics = sorted(metrics, key=lambda x: x[0])
    metrics = [x[1] for x in _metrics]
    assert isinstance(metrics, list)
    res = {k: [] for k in metrics[0].keys()}
    
    for m in metrics:
        to_del = []
        for k in res:
            if k not in m:
                to_del.append(k)
        for k in to_del:
            del res[k]
    
    for m in metrics:
        for k in res.keys():
            res[k].append(m[k])
            
    res_stats = {}
    for k, v in res.items():
        res_stats[k] = {"mean": np.mean(v), "std": np.std(v)}
    logging.info("Raw metrics: {}".format(res))
    return res_stats, metrics


def set_seed(seed, get_state=False, set_torch=True, set_numpy=True):
    if get_state:
        if set_torch:
            torch_state = torch.get_rng_state()
        else:
            torch_state = None
        if set_numpy:
            numpy_state = np.random.get_state()
        else:
            numpy_state = None
    else:
        torch_state, numpy_state = None, None
    if set_torch:
        torch.manual_seed(seed)
    if set_numpy:
        np.random.seed(seed)
    if get_state:
        return (torch_state, numpy_state)


def restore_state(states, set_torch=True, set_numpy=True):
    if set_torch:
        torch.set_rng_state(states[0])
    if set_numpy:
        np.random.set_state(states[1])


class seed_scope:

    def __init__(self, seed, set_torch=True, set_numpy=True):
        self.seed = seed
        self.set_torch = set_torch
        self.set_numpy = set_numpy

    def __enter__(self):
        self.prev_state = set_seed(self.seed, get_state=True,
                                   set_numpy=self.set_numpy,
                                   set_torch=self.set_torch)

    def __exit__(self, type, value, traceback):
        restore_state(self.prev_state,
                      set_torch=self.set_torch,
                      set_numpy=self.set_numpy)


class RandomSplitter:
    def __init__(self, splits, num, seed=None):
        assert np.isclose(np.sum(splits), 1)
        self.num = num
        idx = list(range(num))
        with seed_scope(seed=seed, set_torch=False):
            np.random.shuffle(idx)
        cnt = 0
        self.idx_splits = []
        for s in splits[:-1]:
            length = int(s * num)
            self.idx_splits.append(idx[cnt: cnt+length])
            cnt += length
        self.idx_splits.append(idx[cnt:])

    def split(self, x, split_id=None):
        if split_id is not None:
            return x[self.idx_splits[split_id]]
        else:
            return [x[_idx] for _idx in self.idx_splits]

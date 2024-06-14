import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Callable, TypeVar, Optional

T = TypeVar('T', torch.Tensor, np.ndarray)


def map_structure(data: Union[Tuple[T, ...], Dict[str, T]], f: Callable[[T], T]) -> Union[Tuple[T, ...], Dict[str, T]]:
    if isinstance(data, tuple):
        return tuple(f(d) for d in data)
    elif isinstance(data, dict):
        return {k: f(v) for k, v in data.items()}
    else:
        raise NotImplementedError(type(data))


def cat_structure_np(datas: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    assert isinstance(datas[0], dict), "Not implemented for other types"

    keys = set(datas[0].keys())
    for d in datas[1:]:
        keys.intersection_update(d.keys())

    return {
        k: np.concatenate([d[k] for d in datas]) for k in keys
    }


def stack_structure_np(datas: Tuple[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    assert isinstance(datas[0], dict), "Not implemented for other types"

    keys = set(datas[0].keys())
    for d in datas[1:]:
        keys.intersection_update(d.keys())

    return {
        k: np.stack([d[k] for d in datas]) for k in keys
    }


def flatten_batch(x: torch.Tensor, nonbatch_dims=1) -> Tuple[torch.Tensor, torch.Size]:
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))

    return x, batch_dim


def unflatten_batch(x: torch.Tensor, batch_dim: Union[torch.Size, Tuple]) -> torch.Tensor:
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x

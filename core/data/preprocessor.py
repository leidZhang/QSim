import numpy as np
from typing import Dict, Callable
from torch.utils.data import Dataset, IterableDataset

def remove_keys(data: dict, keys: list):
    for key in keys:
        if key in data:
            del data[key]

def to_image(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
        x = x / 255.0 - 0.5
    else:
        assert 0.0 <= x[0, 0, 0, 0, 0] <= 1.0
        x = x.astype(np.float32)

    x = x.transpose(0, 1, 4, 2, 3) #(T, B, H, W, C) => (T, B, C, H, W)
    return x

class TransformedDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, fn: Callable):
        super().__init__()
        self.dataset = dataset
        self.fn = fn

    def __iter__(self):
        for batch in iter(self.dataset):
            yield self.fn(batch)

class Preprocessor:
    def __init__(
        self,
        image_key="image"
    ):
        self.image_key = image_key

    def __call__(self, dataset: IterableDataset) -> IterableDataset:
        return TransformedDataset(dataset, self.apply)

    def apply(self, batch: Dict[str, np.ndarray], expand_tb=False):
        #expand
        if expand_tb:
            batch = {k: v[np.newaxis, np.newaxis] for k, v in batch.items()}

        # cleanup policy info logged by policy network, not to be confused with current values
        remove_keys(batch, ['policy_value', 'policy_entropy'])

        #image
        if self.image_key:
            batch["image"] = batch[self.image_key]
            batch["image"] = to_image(batch[self.image_key])

        if "action" in batch:
            assert len(batch['action'].shape) == 3 # T, B, action
            batch['action'] = batch['action'].astype(np.float32)

        # reward, terminal
        '''T = batch['reward'].shape[0]
        batch['terminal'] = batch.get('terminal', np.zeros((T,))).astype(np.float32)
        if "reward" in batch:
            batch['reward'] = batch.get('reward', np.zeros((T,))).astype(np.float32)
            #batch['reward'] = clip_rewards_np(batch['reward'], self.clip_rewards)'''

        return batch
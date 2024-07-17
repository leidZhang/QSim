from abc import abstractmethod
from typing import Tuple
from numpy import ndarray
from torch.nn import Module

import torch

from ..templates.base_policy import BasePolicy


class PTPolicy(BasePolicy):
    def __init__(self, model: Module, weight_path: str) -> None:
        self.model: Module = model
        self.model.eval() # Set the model to evaluation mode
        if weight_path is not None:
            model_checkpoint: dict = torch.load(weight_path)
            self.model.load_state_dict(model_checkpoint['model_state_dict'])

    @abstractmethod
    def execute(self, *args) -> Tuple[ndarray, dict]:
        ...

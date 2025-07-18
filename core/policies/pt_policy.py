from abc import abstractmethod
from typing import Tuple
from numpy import ndarray
from torch.nn import Module

import torch

from .base_policy import BasePolicy


class PTPolicy(BasePolicy):
    def __init__(self, model: Module, model_path: str) -> None:
        self.model: Module = model
        self.model.eval() # Set the model to evaluation mode
        model_checkpoint: dict = torch.load(model_path)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])

    @abstractmethod
    def execute(self, *args) -> Tuple[ndarray, dict]:
        ...
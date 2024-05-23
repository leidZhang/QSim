from abc import ABC, abstractmethod

import numpy as np


class EdgeFinder(ABC):
    def __init__(self, image_width: int = 820, image_height: int = 410) -> None:
        self.image_width: int = image_width
        self.image_height: int = image_height

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def execute(self, image: np.ndarray) -> tuple[float, float]:
        ...
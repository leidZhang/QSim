from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class EdgeFinder(ABC):
    """
    EdgeFinder is an abstract class, defines the interface of edge finding classes

    Attributes:
    - image_width: int: width of the frame
    - image_height: int: height of the frame

    Methods:
    - preprocess_image: Preprocess the input image for edge detection. The specific
    preprocessing steps are implemented by subclasses.
    - execute: Execute the edge detection algorithm. The method will be implemented by
    the subclass
    - find_slope_from_binary_image: Find the slope and intercept of the lane from the binary image
    """

    def __init__(self, image_width: int = 820, image_height: int = 410) -> None:
        """
        Initialize the EdgeFinder instance

        Parameters:
        - image_width: int: width of the frame, default value is 820 pixels
        - image_height: int: height of the frame, default value is 410 pixels

        Returns:
        - None
        """
        self.image_width: int = image_width
        self.image_height: int = image_height
            
    @abstractmethod
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesss the input image, including resize, crop, convert to grayscale, etc.
        The specific preprocessing steps are implemented by subclasses.

        Parameters:
        image: np.ndarray: The original image waiting for process

        Returns:
        - np.ndarray: The preprocessed image
        """
        ...

    @abstractmethod
    def execute(self, image: np.ndarray) -> Any:
        """
        Execute the edge detection algorithm

        Parameters:
        - image: np.ndarray: The original image waiting for process

        Returns:
        - Any: Detected edge information, the subclasses will define the format
        """
        ...
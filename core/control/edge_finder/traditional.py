from typing import Tuple, Union

import cv2
import numpy as np

from core.utils.image_utils import find_slope_intercept_from_binary
from .edge_finder import EdgeFinder
from .piplines import SobelPipeLine, ContourPipeLine, HoughPipeLine
from .exceptions import NoImageException


class TraditionalEdgeFinder(EdgeFinder): 
    """
    The TraditionalEdgeFinder class finds the edges of the road using traditional 
    computer vision techniques. It serializes the Hough, Contour, and Sobel pipelines,
    meaning it runs these pipelines in sequence, with the output of one pipeline 
    serving as the input to the next.
    """

    def __init__(self, image_width: int = 820, image_height: int = 410, device: str = 'cpu') -> None:
        """
        Initializes the TraditionEdgeFinder object.

        Parameters:
        - image_width: int: The width of the image
        - image_height: int: The height of the image
        - device: str: The device to run the edge finder on (either 'cpu' or 'gpu')
        """
        self.device: str = device
        if cv2.cuda.getCudaEnabledDeviceCount() < 0 and device == "gpu":
            self.device = "cpu" # if there is no available gpu device, use cpu
        self.preprocess_final_methods: dict = {
            'cpu': self._set_image_for_cpu,
            'gpu': self._set_image_for_gpu,
        }
        super().__init__(image_width=image_width, image_height=image_height)
        self.sobel_pipeline: SobelPipeLine = SobelPipeLine(device=device)
        self.contour_pipeline: ContourPipeLine = ContourPipeLine(device=device)
        self.hough_pipeline: HoughPipeLine = HoughPipeLine(device=device)
        self.image: Union[np.ndarray, cv2.cuda_GpuMat] = None
        self.preprocess_final = self.preprocess_final_methods[self.device]

    def _set_image_for_cpu(self, image: np.ndarray) -> None:
        """
        Sets the image attribute to the input image.

        Parameters:
        - image (np.ndarray): The input image.

        Returns:
        - None
        """
        self.image = image

    def _set_image_for_gpu(self, image: np.ndarray) -> None:
        """
        Sets the image attribute to the input image and uploads it to the GPU.

        Parameters:
        - image (np.ndarray): The input image.

        """
        self.image = cv2.cuda_GpuMat()
        self.image.upload(image)
    
    def _preprocess_image(self, image: np.ndarray) -> None:
        """
        Preprocesses the input image for line following.

        Returns:
        - np.ndarray: Preprocessed image for line following.

        Raises:
        - NoImageException: If the input image is None.
        """
        # check if the image is None
        if image is None:
            raise NoImageException()
        # crop the image
        local_image = image[230:360, 100:]
        # resize the image
        local_image = cv2.resize(local_image, (local_image.shape[1] // 2, local_image.shape[0] // 2))
        # convert the image to grayscale
        local_image: np.ndarray = cv2.cvtColor(local_image, cv2.COLOR_BGR2GRAY)
        self.preprocess_final(local_image)

    def execute(self, original_image: np.ndarray) -> Tuple[float, float]:
        """
        Executes the traditional edge finder.

        Parameters:
        - original_image (np.ndarray): Original image for line following.

        Returns:
        - Tuple: The slope and intercept of the detected edge.
        """
        self._preprocess_image(original_image)
        self.hough_pipeline(self.image) # draw the hough lines on the image
        largest_contour: np.ndarray = self.contour_pipeline(self.image)
        edge: np.ndarray = self.sobel_pipeline(largest_contour, self.image)
        result: Tuple[float, float] = find_slope_intercept_from_binary(binary=edge)
        return (result[0] - 0.02, result[1] * 2 + 1)

import os

import cv2
import numpy as np


class ImageWriter:
    def __init__(self, output_path: str) -> None:
        self.output_path: str = output_path
        self.images: list = []
        self.counter: int = 0
        # check if the output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def add_image(self, image: np.ndarray) -> None:
        self.images.append(image.copy())

    def write_images(self) -> None:
        for i in range(len(self.images)):
            image_path = os.path.join(self.output_path, f"image_{self.counter // 2}.jpg")
            cv2.imwrite(image_path, self.images[i])
            self.counter += 1
        self.images = []


class ImageReader:
    def __init__(self, input_path: str) -> None:
        self.input_path: str = input_path
        self.images: list = self._get_image_files()
    
    def _get_image_files(self) -> list:
        # Get all files in the input path
        files = os.listdir(self.input_path)
        # Filter out the files that are not images
        image_files = [file for file in files if os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']]
        image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return image_files
    
    def read_images(self) -> list:
        for image_file in self.images:
            image_path = os.path.join(self.input_path, image_file)
            image = cv2.imread(image_path)
            cv2.imshow('Image', image)
            cv2.waitKey(30)

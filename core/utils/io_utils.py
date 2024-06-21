import os
import json
from typing import Any

import cv2
import numpy as np

from .performance import skip


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


class DataWriter:
    def __init__(self, folder_path: str, file_path: str) -> None:
        self.history: list = []
        self.process_data = skip
        self.folder_path: str = os.path.join(
            os.getcwd(), folder_path
        )
        # check if the output path exists
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        # check if the file path exists
        self.path: str = os.path.join(self.folder_path, file_path)
        if not os.path.exists(self.path):
            open(self.path, 'w').close()
        
    def add_data(self, data: Any) -> None:
        self.history.append(data.copy())

    def _write_to_json(self, file) -> None:
        for i in len(self.history):
            data: Any = self.history[i]
            self.process_data(data, i)
            json.dump(data, file, indent=4)
        file.write('\n')

    def write_data(self) -> None:
        with open(self.path, 'w') as f:
            self._write_to_json(f)
        self.history = []
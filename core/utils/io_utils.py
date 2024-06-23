import os
import json
from abc import ABC, abstractmethod
from typing import Any, Union, List

import cv2
import numpy as np

from .performance import skip


def convert_ndarray_to_int_list(data: np.ndarray) -> List[int]:
    data = data.tolist()
    return [int(data[i]) for i in range(len(data))]


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


class DataWriter(ABC):
    def __init__(self, folder_path: str) -> None:
        self.history: list = []
        self.process_data = skip
        self.folder_path: str = os.path.join(
            os.getcwd(), folder_path
        )
    
    def add_data(self, data: Any) -> None:
        self.history.append(data)

    @abstractmethod
    def write_data(self) -> None:
        ...


class JSONDataWriter(DataWriter):
    def __init__(self, folder_path: str) -> None:
        super().__init__(folder_path)
        # check if the output path exists
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            os.makedirs(os.path.join(self.folder_path, "jsons"))
        
    def _write_to_json(self) -> None:
        # initialize the episode data
        timestamp: str = self.history[0]['timestamp']
        task: str = self.history[0]['task']
        task_length: int = self.history[0]['task_length']
        filename: str = os.path.join(
            self.folder_path, "jsons", f"episode_{timestamp}.json"
        )
        episode_data = {
            "timestamp": timestamp,
            "task": task,
            "task_length": task_length,
            "steps": [],
        }
        # process the data
        history_len = len(self.history)
        for i in range(history_len):
            data: dict = self.history.pop(0)
            self.process_data(data, i)
            episode_data["steps"].append(data)
        # save the data to the json file
        with open(filename, "w") as f:
            json.dump(episode_data, f)
    
    def write_data(self) -> None:
        if len(self.history) > 0:
            self._write_to_json()
            self.history = []
            print("Data written to json file!")


class NPZDataWriter(DataWriter):
    def __init__(self, folder_path: str) -> None:
        super().__init__(folder_path)
        # check if the output path exists
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            os.makedirs(os.path.join(self.folder_path, "npzs"))

    def _write_to_npz(self) -> None:
        # initialize the episode data
        timestamp: str = self.history[0]['timestamp']
        task: str = self.history[0]['task']
        filename: str = os.path.join(
            self.folder_path, "npzs", f"episode_{timestamp}.npz"
        )
        episode_data = {
            "timestamp": timestamp,
            "task": task,
            "steps": [],
        }
        # process the data
        for i in range(len(self.history)):
            data: dict = self.history[i]
            self.process_data(data, i)
            episode_data["steps"].append(data)
        # save the data to the npz file
        np.savez(filename, **episode_data)

    def write_data(self) -> None:
        if len(self.history) > 0:
            self._write_to_npz()
            self.history = []
            print("Data written to npz file!")
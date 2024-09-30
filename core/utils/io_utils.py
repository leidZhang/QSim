import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Union, List, Callable, Dict
from multiprocessing import Queue

import cv2
import numpy as np

from .performance import skip


def convert_ndarray_to_int_list(data: np.ndarray) -> List[int]:
    data = data.tolist()
    return [int(data[i]) for i in range(len(data))]


def read_npz_file(file_path: str) -> Union[dict, None]:
    data: dict = np.load(file_path)
    try:
        data['sentinel'] # test the data integrity
    except Exception as e:
        print(f"{file_path} Corrupted!")
        return None
    return data


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
        self.process_data: Callable = skip
        self.get_episode_data: Callable = skip
        self.folder_path: str = os.path.join(
            os.getcwd(), folder_path
        )

    def add_data(self, data: Any) -> None:
        self.history.append(data)

    def _get_timestamped_filename(self, extension: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return os.path.join(self.folder_path, f"episode_{timestamp}.{extension}")

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

    def _write_to_json(self, *args: Any) -> None:
        filename = self._get_timestamped_filename("json")
        episode_data: Dict[str, Any] = self.get_episode_data(*args)
        try:
            with open(filename, "w") as f:
                json.dump(episode_data, f)
            print(f"Data written to {filename}")
        except IOError as e:
            print(f"Failed to write JSON data: {e}")

    def write_data(self, folder_path: str) -> None:
        if len(self.history) > 0:
            self._write_to_json(folder_path)
            self.history = []


class NPZDataWriter(DataWriter):
    def __init__(self, folder_path: str) -> None:
        super().__init__(folder_path)
        # check if the output path exists
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            os.makedirs(os.path.join(self.folder_path, "npzs"))

    def _write_to_npz(self, *args: Any) -> None:
        filename = self._get_timestamped_filename("npz")
        episode_data: Dict[str, Any] = self.get_episode_data(*args)
        episode_data["sentinel"] = True  # for data integrity checking
        try:
            with open(filename, 'wb') as f:
                np.savez(f, **episode_data)
            print(f"Data written to {filename}")
        except IOError as e:
            print(f"Failed to write NPZ data: {e}")

    def write_data(self) -> None:
        if len(self.history) > 0:
            self._write_to_npz()
            self.history = []

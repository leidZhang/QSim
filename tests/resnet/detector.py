import time
from typing import Tuple

import torch
import torch.nn as nn
import cv2
import numpy as np

from tests.resnet.model import ObstacleDetection


class ResNetDetector:
    def __init__(self, model: nn.Module, weights_path: str) -> None:
        self.model = model
        self.model.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()

    def __call__(self, img: np.ndarray) -> Tuple[float, float]:
        # preprocess image
        preprocessed_image = cv2.resize(img, (224, 224))
        preprocessed_image = torch.from_numpy(preprocessed_image).float().cuda() / 255.
        preprocessed_image = preprocessed_image.permute(2, 0, 1)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        # prediction
        cls_pred, dis_pred = self.model(preprocessed_image)
        return cls_pred.item(), dis_pred.item()
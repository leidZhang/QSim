import os
import cv2
import numpy as np
from typing import List
from system.settings import NPZ_DIR

from tests.test_dataset import test_online_mean_and_std_2
from tests.test_model import test_model_next_obs


if __name__ == "__main__":
    # test_online_mean_and_std_2()
    # test_model_next_obs()
    file_dir: str = os.path.join(NPZ_DIR, "9834922e-63dc-11ef-b0fd-01919009f363_agent.npz")
    with np.load(file_dir) as data:
        for key in data:
            print(key)
            print(data[key].shape)

        images: np.ndarray = data["image"]
        action: np.ndarray = data["action"]
        reward: np.ndarray = data["reward"]
        interventions: np.ndarray = data["intervention"]
        print(len(reward))
        print(len(images))
        for i in range(len(images)):
            # print("Reward:", reward[i])
            print("Action:", action[i])
            # print("Intervention:", interventions[i])
            cv2.imshow("Image", images[i])
            cv2.waitKey(100)
        cv2.destroyAllWindows()
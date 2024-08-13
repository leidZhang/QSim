import os
import cv2
import numpy as np
from typing import List
from system.settings import NPZ_DIR

if __name__ == "__main__":
    file_dir: str = os.path.join(NPZ_DIR, "b967d86e-59c2-11ef-803f-01914dd730e9.npz")
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
            print("Reward:", reward[i])
            # print("Action:", action[i])
            # print("Intervention:", interventions[i])
            cv2.imshow("Image", images[i])
            cv2.waitKey(30)
        cv2.destroyAllWindows()
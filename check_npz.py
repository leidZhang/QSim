import os
import glob

import cv2
import numpy as np
from typing import List
from system.settings import NPZ_DIR

from restful.repository import DatasetRepository


def check_npz():
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


if __name__ == "__main__":
    # test_online_mean_and_std_2()
    # test_model_next_obs()
    repo: DatasetRepository = DatasetRepository()
    npz_dir_list: List[str] = glob.glob(os.path.join(NPZ_DIR, "*.npz"))

    # path = r"C:\Users\sdcnlab025\Desktop\HaoZhang\HITL_Reinformer\state_relay\assets/npz\0005ed77-64a5-11ef-9ee9-0191952b5416_agent.npz"    
    # for npz_dir in npz_dir_list:
    #     data = repo.read_from_npz(npz_dir)
    #     repo.save_to_db(npz_dir, data)

    # for i, npz_dir in enumerate(npz_dir_list):
    #     data = repo.read_from_npz(npz_dir)
    #     length = len(data["reward"])
    #     print("episode", i, data["reward"][length - 1])



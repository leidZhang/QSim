import os
import json

import cv2
import torch
import numpy as np

from reinformer.model import ReinFormer
from reinformer.vehicle import ReinformerPolicy
from reinformer.settings import *


class TestPolicy(ReinformerPolicy):
    def execute(self, observation: dict) -> None:
        observation_shape = self.states[0, self.step_counter].shape
        self.states[0, self.step_counter] = torch.from_numpy(observation['observations'].reshape(observation_shape)).to(self.device)
        self.states[0, self.step_counter] = (self.states[0, self.step_counter] - self.state_mean) / self.state_std
        if self.step_counter < self.context_len:
            _, action_predict, state_predict = self.model.forward(
                self.timesteps[:, :self.context_len],
                self.states[:, :self.context_len],
                self.actions[:, :self.context_len],
                self.returns_to_go[:, :self.context_len],
            )
        else:
            _, action_predict, state_predict = self.model.forward(
                self.timesteps[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
                self.states[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
                self.actions[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
                self.returns_to_go[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
            )
        # print("Action: ", action_predict.scale)
        # print("Next state", state_predict.shape)

        action: torch.Tensor = action_predict.mean.reshape(1, -1, self.act_dim)[0, -1].detach()
        next_state: torch.Tensor = state_predict.squeeze(0).mean(dim=0).detach()
        self.step_counter += 1
        # print(state_predict)
        return action.cpu().numpy(), next_state.cpu().numpy()

def test_model_next_obs() -> None:
    npz_path: str = r"assets/test_npz/0a1f8616-5b42-11ef-84d2-019157a74946_agent.npz"
    weight_path: str = r"assets/models/backup_20240822144157.pt"
    data: dict = np.load(npz_path, allow_pickle=True)

    model: ReinFormer = ReinFormer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        n_blocks=4,
        h_dim=EMBED_DIM,
        context_len=CONTEXT_LEN,
        n_heads=2,
        drop_p=0.1,
        init_temperature=0.1,
        target_entropy=-ACT_DIM
    )

    with open("state_stat.json", "r") as f:
        state_stat: dict = json.load(f)
    STATE_MEAN: np.ndarray = np.array(state_stat["state_mean"])
    STATE_STD: np.ndarray = np.array(state_stat["state_std"])
    policy: ReinformerPolicy = TestPolicy(model, weight_path)
    policy.setup(
        eval_batch_size=1,
        max_test_ep_len=1000,
        context_len=CONTEXT_LEN,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        device="cpu"
    )

    for i in range(len(data["image"]) - 1):
        with torch.no_grad():
            image: np.ndarray = cv2.resize(data["image"][i], (84, 84))
            next_image: np.ndarray = cv2.resize(data["image"][i+1], (84, 84))
            state: np.ndarray = image.reshape(-1)
            state = np.concatenate((state, data["state_info"][i]))
            observation: dict = {
                "observations": state,
                "action": data["action"][i],
                "image": image,
            }
            action, next_state = policy.execute(observation)
            next_state = next_state[:-6].reshape((84, 84, 3))

            cv2.namedWindow("Real_Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Predicted", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Real_Image", 500, 500)
            cv2.resizeWindow("Predicted", 500, 500)

            cv2.imshow("Real_Image", next_image)
            cv2.imshow("Predicted", next_state)
            cv2.waitKey(30)

            print(f"Action: {action}")

    
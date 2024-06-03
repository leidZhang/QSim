import torch
import numpy as np

from core.models.torch.model import Model
from core.data.preprocessor import Preprocessor
from core.utils.agg_utils import map_structure

class NetworkPolicy:
    def __init__(self, device):
        self.device = device
        self.model = Model(device=device)
        self.preprocessor = Preprocessor(
            image_key="image"
        )
        self.recurrent_state = None

        self.mins = np.array([-0.05, -0.5])
        self.maxs = np.array([0.2, 0.5])

    def __call__(self, obs, steps):
        metrics = {}
        batch = self.preprocessor.apply(obs, expand_t=True)
        obs_model: Dict[str, torch.Tensor] = map_structure(map_structure(batch, torch.from_numpy), lambda x: x.to(self.device))  # type: ignore

        action, recurrent_state, model_metrics = self.model.inference(obs_model, self.recurrent_state)
        metrics.update(model_metrics)
        self.recurrent_state = recurrent_state

        action = action.squeeze().cpu().numpy()
        action_noise = (0.9999**steps) * np.random.normal(loc=np.array([0.05, 0.0]), scale=np.array([0.2, 0.25])) 
        #print("ACTION")
        #print(action)
        #print(action_noise)

        action = np.clip(np.clip(action, self.mins, self.maxs) + action_noise, self.mins, self.maxs)

        return action, metrics

    def reset_state(self):
        self.recurrent_state = self.model.init_state()
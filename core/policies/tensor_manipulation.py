import torch
action_tensor = torch.from_numpy(action)
action = (torch.randn_like(action_tensor) * 0.02).numpy()
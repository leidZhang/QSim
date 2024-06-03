import torch
import torch.nn as nn

from core.models.torch.common import MLP
from core.utils.agg_utils import flatten_batch, unflatten_batch

class DenseDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        hidden_layers,
        layer_norm=nn.LayerNorm,
        activation=nn.ReLU
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation)
        self.criterion = torch.nn.SmoothL1Loss()

    def forward(self, features):
        y = self.model.forward(features)
        return y

    def loss(self, prediction, waypoints):
        prediction = prediction.reshape(-1, self.out_dim // 2, 2)
        loss = self.criterion(prediction, waypoints)
        return loss

class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_channels = 3,
        depth: int = 32,
        activation = nn.ELU
    ):
        super().__init__()
        self.in_dim = in_dim

        layers = [
            nn.Linear(in_dim, 8964),
            nn.Unflatten(-1, (747, 3, 4)),
            nn.ConvTranspose2d(747, 64, kernel_size=4, stride=3),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=3),
            activation(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2, padding=(2, 0))
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, prediction, target):
        return torch.square(prediction - target).sum([-1, -2, -3])
import torch
import torch.nn as nn
from typing import Dict

#project imports
from core.utils.agg_utils import flatten_batch, unflatten_batch

class MultiEncoder(nn.Module):
    def __init__(
        self, 
        include_image=True,
        include_vector=False
    ):
        super().__init__()
        self.include_image = include_image
        self.include_vector = include_vector

        if self.include_image:
            self.image_encoder = ConvEncoder(in_channels = 3, depth=32)
        else:
            self.image_encoder = None

        #not implemented yet
        if self.include_vector:
            self.vector_encoder = None
        else:
            self.vector_encoder = None

        self.out_dim = ((self.image_encoder.out_dim if self.image_encoder else 0) + 
                        (self.vector_encoder.out_dim if self.vector_encoder else 0))

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeds = []

        #get image embeddings
        if self.image_encoder:
            image = obs["image"]
            #image = flatten_batch(obs["image"], nonbatch_dims=3)[0]

            embed_image = self.image_encoder.forward(image)
            embeds.append(embed_image)

        #get vectorb observation embeddings
        if self.vector_encoder:
            embed_vecobs = self.vector_encoder(obs["vecobs"])
            embeds.append(embed_vecobs)

        embed = torch.cat(embeds, dim=-1)

        return embed

class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 32,
        activation = nn.ELU
    ):
        super().__init__()

        self.out_dim = 8960

        #Convolutional layers
        '''self.conv_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )'''

        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.Flatten()
        )

        #Dense linear layers
        self.dense_model = nn.Sequential(  
            #nn.Linear(128 * 30 * 40, 512),
            nn.Linear(128 * 10 * 7, 512),
            activation(),
            nn.Linear(512, 256),
            activation(),
            nn.Linear(256, 128)
        )

        '''super().__init__()
        self.out_dim = depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2

        d = in_channels
        hidden_dim = depth
        layers = []
        for i in range(len(kernels)):
            layers += [
                nn.Conv2d(d, hidden_dim, kernels[i], stride),
                activation()
            ]
            d = hidden_dim
            hidden_dim *= 2

        layers += [nn.Flatten()]
        self.model = nn.Sequential(*layers)'''

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.conv_model(x)
        #print("SHAPE")
        #print(x.shape)

        #y = self.dense_model(y)
        y = unflatten_batch(y, bd)
        return y


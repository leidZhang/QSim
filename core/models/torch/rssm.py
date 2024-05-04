import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
import torch.distributions as D

#project imports
from core.models.torch.rnn import GRUCellStack

class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim,
        action_dim,
        deter_dim,
        stoch_dim,
        stoch_discrete,
        hidden_dim,
        gru_layers,
        gru_type,
        layer_norm=True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.hidden_dim = hidden_dim

        self.cell = RSSMCell(
            embed_dim,
            action_dim,
            deter_dim,
            stoch_dim,
            stoch_discrete,
            hidden_dim,
            gru_layers,
            gru_type,
            layer_norm
        )

    def forward(
        self,
        embeds: torch.Tensor, #(T, B, F)
        action: torch.Tensor, #(T, B, A)
        reset: torch.Tensor, #(T, B, R)
        in_state: torch.Tensor,
        do_open_loop=False
    ):
        T, B, A = embeds.shape[:3]
        embeds = embeds.unbind(0) #(T, B, F) => List[(B, F), ...]
        actions = action.unbind(0)
        reset_masks = (~reset.unsqueeze(-1)).unbind(0)

        priors = []
        posts = []
        states_h = []
        samples = []
        (h, z) = in_state
        for i in range(T):
            if not do_open_loop:
                post, (h, z) = self.cell.forward(embeds[i], actions[i], reset_masks[i], (h, z))
            else:
                post, (h, z) = self.cell.forward(actions[i], reset_masks[i], (h, z))

            posts.append(post)
            states_h.append(h)
            samples.append(z)

        #following shapes: (T, B, X)
        posts = torch.stack(posts)
        states_h = torch.stack(states_h) 
        samples = torch.stack(samples) 
        priors = self.cell.batch_prior(states_h) 
        features = self.to_feature(states_h, samples)
        states = (states_h, samples)

        return (
            priors,
            posts,
            samples,
            features,
            states,
            (h.detach(), z.detach())
        )

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat((h, z), -1)
    
    def zdistr(self, pp: torch.Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)

class RSSMCell(nn.Module):
    def __init__(
        self,
        embed_dim,
        action_dim,
        deter_dim,
        stoch_dim,
        stoch_discrete,
        hidden_dim,
        gru_layers,
        gru_type,
        layer_norm=True
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.layer_norm = layer_norm

        #input latent and action layers
        self.z_layer = nn.Linear(stoch_dim * (stoch_discrete or 1), hidden_dim)
        self.a_layer = nn.Linear(action_dim, hidden_dim, bias=False)
        self.in_norm = nn.LayerNorm(hidden_dim, eps=1e-3) if layer_norm else None

        #recurrent module
        self.gru = GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)

        #prior layers
        self.prior_layer_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = nn.LayerNorm(hidden_dim, eps=1e-3) if layer_norm else None
        self.prior_layer = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

        #posterior layers
        self.post_layer_h = nn.Linear(deter_dim, hidden_dim)
        self.post_layer_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-3) if layer_norm else None
        self.post_layer = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

    def init_state(self, bacth_size):
        return (
            torch.zeros((bacth_size, self.deter_dim), device=self.device),
            torch.zeros((bacth_size, self.stoch_dim * (self.stoch_discrete or 1)), device=self.device)
        )

    def forward(
        self,
        embed: torch.Tensor, # (BA, X)
        action: torch.Tensor, # (BA, X)
        reset_mask: torch.Tensor, # (BA)
        in_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        in_h, in_z = in_state

        print("SHAPES")
        print(in_h.shape)
        print(in_z.shape)
        print(embed.shape)
        print(action.shape)
        print(reset_mask.shape)

        in_h = in_h * reset_mask
        in_z = in_z * reset_mask
        batch_size = action.shape[0]

        x = self.z_layer(in_z) + self.a_layer(action) #(BA, H)
        x = self.in_norm(x) if self.in_norm is not None else x
        za = F.elu(x)

        h = self.gru(za, in_h)

        x = self.post_layer_h(h) + self.post_layer_e(embed)
        x = self.post_norm(x) if self.post_norm is not None else x
        post_in = F.elu(x)
        post = self.post_layer(post_in)
        dist = self.zdistr(post)
        sample = dist.rsample().reshape(batch_size, -1)

        return (post, (h, sample))

    def forward_prior(
        self,
        action: torch.Tensor,
        reset_mask: torch.Tensor,
        in_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask
        batch_size = action.shape[0]

        x = self.z_layer(in_z) + self.a_layer(action)
        x = self.in_norm(x) if self.in_norm is not None else x
        za = F.elu(x)
        h = self.gru(za, in_h)

        x = self.prior_layer_h(h)
        x = self.prior_norm(x) if self.prior_norm is not None else x
        x = F.elu(x)
        prior = self.prior_layer(x)
        dist = self.zdistr(prior)
        sample = dist.rsample().reshape(batch_size, -1)

        return (prior, (h, sample))

    def batch_prior(
        self,
        h: torch.Tensor
    ) -> torch.Tensor:
        x = self.prior_layer_h(h)
        x = self.prior_norm(x) if self.prior_norm is not None else x
        x = F.elu(x)
        prior = self.prior_layer(x)

        return prior

    def zdistr(self, pp: torch.Tensor, max_std=2.0, min_std=0.1) -> D.Distribution:
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            dist = D.OneHotCategoricalStraightThrough(logits=logits.float())
            dist = D.independent.Independent(dist, 1)
        else:
            mean, std = pp.chunk(2, -1)
            std = (max_std - min_std) * torch.sigmoid(std) + min_std
            dist = D.normal.Normal(mean, std)
            dist = D.independent.Independent(dist, 1)

        return dist
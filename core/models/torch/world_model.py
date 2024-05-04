import torch
import torch.nn as nn
import torch.distributions as D
from typing import Any, Tuple, Dict

#project imports
from core.models.torch.rssm import RSSM
from core.models.torch.a2c import ActorCritic
from core.models.torch.rnn import GRUCellStack
from core.models.torch.encoders import MultiEncoder
from core.models.torch.decoders import DenseDecoder, ConvDecoder
from core.utils.aggregation_utils import flatten_batch

class WorldModel(nn.Module):
    def __init__(
        self,
        deter_dim = 1024,
        stoch_dim = 16,
        stoch_discrete = 16,
        device="cuda:0"
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.device = device

        self.encoder = MultiEncoder(
            include_image=True
        ).to(self.device)

        self.rssm = RSSM(
            embed_dim = self.encoder.out_dim,
            action_dim = 2,
            deter_dim = deter_dim,
            stoch_dim = stoch_dim,
            stoch_discrete = stoch_discrete,
            hidden_dim = 1000,
            gru_layers = 1,
            gru_type = "gru"
        ).to(self.device)

        self.decoder = ConvDecoder(
            in_dim = self.rssm.deter_dim + (self.rssm.stoch_dim * self.rssm.stoch_discrete)
        ).to(self.device)

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        optimizer_encoder = torch.optim.AdamW(self.encoder.parameters(), lr=lr, eps=eps)
        optimizer_decoder = torch.optim.AdamW(self.decoder.parameters(), lr=lr, eps=eps)
        return optimizer_encoder, optimizer_decoder

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        grad_metrics = {
            'grad_norm_encoder': nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_clip),
            'grad_norm_decoder': nn.utils.clip_grad_norm_(self.decoder.parameters(), grad_clip)
        }
        return grad_metrics

    def init_state(self, batch_length: int = 1, batch_size: int = 1):
        return (
            torch.zeros((batch_size, self.deter_dim), device=self.device),
            torch.zeros((batch_size, self.stoch_dim * self.stoch_discrete), device=self.device)
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        in_state: Any
    ):
        metrics = {}
        pred = {}

        embed = self.encoder(obs)
        #embed, _ = flatten_batch(embed)
        prior, post, samples, features, states, out_state = self.rssm(
            embed,
            obs["action"],
            obs["reset"],
            in_state
        )
        features, _ = flatten_batch(features)
        image = self.decoder(features)
        
        with torch.no_grad():
            pred["image"] = image.detach()

        return pred, out_state, metrics

    def training_step(
        self,
        obs: Dict[str, torch.Tensor],
        in_state: Any
    ):
        metrics = {}
        pred = {}

        embed = self.encoder(obs)
        #embed, _ = flatten_batch(embed)

        prior, post, samples, features, states, out_state = self.rssm(
            embed,
            obs["action"],
            obs["reset"],
            in_state
        )

        #get distributions of prior and posterior samples
        # we want the prior to move towards the representation model
        # but we also want the representations to be regularized towards the prior
        # of course we want to insentivize the piors more, so we addmore weight to those
        dprior = self.rssm.zdistr(prior)
        dpost = self.rssm.zdistr(post)
        kl_loss_post_grad = D.kl.kl_divergence(dpost, self.rssm.zdistr(prior.detach())) #kl divergence with prior gradients removed (posterior moves towards prior in backprop)
        kl_loss_prior_grad = D.kl.kl_divergence(self.rssm.zdistr(post.detach()), dprior) #kl divergence with posterior gradients removed (prior moves towards posterior in backprop)
        kl_loss = (0.2 * kl_loss_post_grad) + (0.8 * kl_loss_prior_grad)

        print("FEATURES")
        print(features.shape)
        features, _ = flatten_batch(embed)

        image_pred = self.decoder(features)
        loss_image = self.decoder.loss(image_pred, flatten_batch(obs["image"], nonbatch_dims=3)[0])

        loss = torch.stack([
            loss_image.mean(),
            kl_loss.mean()
        ]).mean()

        with torch.no_grad():
            metrics.update({
                "loss_image": loss_image.mean().detach(),
                "loss_transition": kl_loss.mean().detach(),
                "loss": loss.detach()
            })

        return loss, metrics
import torch
import torch.nn as nn
from typing import Any, Tuple, Dict

#project imports
from core.models.torch.a2c import ActorCritic
from core.models.torch.rnn import GRUCellStack
from core.models.torch.encoders import MultiEncoder
from core.models.torch.decoders import DenseDecoder
from core.utils.agg_utils import flatten_batch

class Model(nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = device
        self.gru_hidden_dim = 1024

        self.encoder = MultiEncoder(
            include_image=True
        ).to(self.device)

        self.gru = GRUCellStack(
            input_size = self.encoder.out_dim, 
            hidden_size = self.gru_hidden_dim, 
            num_layers = 1, 
            cell_type = "gru"
        ).to(self.device)

        self.waypoint_decoder = DenseDecoder(
            in_dim=self.gru_hidden_dim,
            out_dim=398,
            hidden_dim=400,
            hidden_layers=2
        ).to(self.device)

        self.waypoint_anchor_decoder = DenseDecoder(
            in_dim=self.gru_hidden_dim,
            out_dim=2,
            hidden_dim=400,
            hidden_layers=2
        ).to(self.device)

        '''self.controller = ActorCritic(
            in_dim=1024,
            out_actions=2,
            hidden_dim = 400,
            hidden_layers = 4,
            layer_norm=True,
            gamma=0.999,
            lambda_gae=0.95,
            entropy_weight=1e-3,
            target_interval=100,
            device = self.device
        ).to(self.device)'''

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        optimizer_encoder = torch.optim.AdamW(self.encoder.parameters(), lr=lr, eps=eps)
        optimizer_decoder = torch.optim.AdamW(self.waypoint_decoder.parameters(), lr=lr, eps=eps)
        #optimizer_rnn = torch.optim.AdamW(self.gru.parameters(), lr=lr, eps=eps)
        #optimizer_actor = torch.optim.AdamW(self.controller.actor.parameters(), lr=lr_actor or lr, eps=eps)
        #optimizer_critic = torch.optim.AdamW(self.controller.critic.parameters(), lr=lr_critic or lr, eps=eps)
        return optimizer_encoder, optimizer_decoder

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        grad_metrics = {
            'grad_norm_encoder': nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_clip),
            'grad_norm_decoder': nn.utils.clip_grad_norm_(self.waypoint_decoder.parameters(), grad_clip),
            #'grad_norm_rnn': nn.utils.clip_grad_norm_(self.gru.parameters(), grad_clip),
            #'grad_norm_actor': nn.utils.clip_grad_norm_(self.controller.actor.parameters(), grad_clip_ac or grad_clip),
            #'grad_norm_critic': nn.utils.clip_grad_norm_(self.controller.critic.parameters(), grad_clip_ac or grad_clip),
        }
        return grad_metrics

    def init_state(self, batch_length: int = 1, batch_size: int = 1):
        return (
            torch.zeros((batch_length, batch_size, self.gru_hidden_dim), device=self.device),
            torch.zeros((batch_length, batch_size, 1,), device=self.device)
        )

    def inference(
        self,
        obs: Dict[str, torch.Tensor],
        in_state: Any
    ):
        metrics = {}

        in_h, in_z = in_state
        in_h, _ = flatten_batch(in_h)
        embed = self.encoder(obs)
        embed, _ = flatten_batch(embed)

        h = self.gru(embed, in_h)
        z = torch.zeros((1,), device=self.device)

        anchor_pred = self.waypoint_anchor_decoder(h).reshape(1, 2)
        waypoint_pred = self.waypoint_decoder(h).reshape(self.waypoint_decoder.out_dim // 2, 2)
        waypoint_pred = torch.vstack([anchor_pred, waypoint_pred])
        waypoint_pred = torch.cumsum(waypoint_pred, dim=0)

        

        '''dist = self.controller.forward_actor(h)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.controller.forward_value(h)'''

        #with torch.no_grad():
        #    metrics["log_prob"] = log_prob.cpu().numpy()
        #    metrics["value"] = value.cpu().numpy()

        #return waypoint_pred, metrics
        return waypoint_pred, (h, z), metrics

    def training_step(
        self,
        obs: Dict[str, torch.Tensor]
    ):
        metrics = {}
        h, z = self.init_state(1, 4)
        horizon = obs["reward"].shape[0]
        embed = self.encoder(obs)

        #print("EMBED SHAPE")
        #print(embed.shape)
        #print(h.shape)

        feats = []
        for i in range(horizon):
            h = self.gru(embed[i], h.squeeze(0))
            feats.append(h)
        
        feats = torch.stack(feats)

        #print("FEATS")
        #print(feats.shape)

        '''loss_actor, loss_critic, entropy = self.controller.training_step(
            feats,
            obs["action"],
            obs["reward"],
            obs["terminal"]
        )'''

        waypoints_gt = flatten_batch(obs["waypoints"], nonbatch_dims=2)[0]
        waypoint_diff = torch.diff(waypoints_gt, dim=1)

        anchor_pred = self.waypoint_anchor_decoder(feats).reshape(-1, 1, 2)
        waypoint_pred = self.waypoint_decoder(feats).reshape(-1, self.waypoint_decoder.out_dim // 2, 2)
        loss_anchor = 1000 * self.waypoint_anchor_decoder.loss(anchor_pred, waypoints_gt[:, [0]])
        loss_waypoints = 1000 * self.waypoint_decoder.loss(waypoint_pred, waypoint_diff)

        loss = torch.stack([
            loss_waypoints,
            loss_anchor
        ]).mean()

        with torch.no_grad():
            metrics.update({
                "loss_waypoints": loss_waypoints.detach(),
                "loss_anchor": loss_anchor.detach(),
                "loss": loss.detach()
            })

        return loss, metrics

        

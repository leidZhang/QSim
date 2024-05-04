import torch
import torch.nn as nn
import torch.distributions as D

#project imports
from core.models.torch.common import MLP

class ActorCritic(nn.Module):
    def __init__(
        self,
        in_dim,
        out_actions,
        hidden_dim = 400,
        hidden_layers = 4,
        layer_norm=True,
        gamma=0.999,
        lambda_gae=0.95,
        entropy_weight=1e-3,
        target_interval=100,
        device="cuda:0"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.entropy_weight = entropy_weight
        self.target_interval = target_interval
        self.device = device

        actor_out_dim = 2 * self.out_actions #mean and stdev of distribution
        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target.requires_grad_(False)
        self.train_steps = 0

    def forward(
        self, 
        features: torch.Tensor
    ):
        return x

    def forward_actor(
        self,
        features: torch.Tensor,
        min_std: float = 0.01,
        max_std: float = 0.05
    ) -> D.Distribution:
        x = self.actor.forward(features)

        mean_, std_ = x.chunk(2, -1)
        #mean = torch.tensor([0.2, 0.5], device=self.device) * torch.tanh(mean_ / torch.tensor([0.2, 0.5], device=self.device))
        mean = torch.tensor([0.1], device=self.device) * torch.tanh(mean_ / torch.tensor([0.1], device=self.device))
        std = max_std * torch.sigmoid(std_) + min_std
        normal = D.normal.Normal(mean, std)
        normal = D.independent.Independent(normal, 1)

        return normal

    def forward_value(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        y = self.critic.forward(features)
        return y

    def training_step(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        cum_rewards = torch.zeros_like(rewards, device=self.device)
        reward_len = rewards.shape[0]
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*self.gamma if j+1 < reward_len else 0)

        values = self.critic.forward(features)
        loss_critic = (0.5 * torch.square(values - cum_rewards)).mean()

        with torch.no_grad():
            values = self.critic.forward(features)

        advantages = cum_rewards - values
        policy_distr = self.forward_actor(features)
        action_logprob = policy_distr.log_prob(actions)
        loss_policy = -action_logprob * advantages

        policy_entropy = policy_distr.entropy()
        loss_actor = (loss_policy - self.entropy_weight * policy_entropy).mean()

        '''if self.train_steps % self.target_interval == 0:
            self.update_critic_target()

        reward1 = rewards[1:]
        terminal0 = terminals[:-1]
        terminal1 = terminals[1:]

        value_t = self.critic.forward(features)
        value0t = value_t[:-1]
        value1t = value_t[1:]
        advantage = - value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
        advantage_gae = []
        agae = None
        for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
            if agae is None:
                agae = adv
            else:
                agae = adv + self.lambda_ * self.gamma * (1.0 - term) * agae
            advantage_gae.append(agae)
        advantage_gae.reverse()
        advantage_gae = torch.stack(advantage_gae)
        value_target = advantage_gae + value0t

        #ignore losses after terminal state
        reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()

        #Critic loss
        value: TensorJM = self.critic.forward(features)
        value0: TensorHM = value[:-1]
        loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        loss_critic = (loss_critic * reality_weight).mean()

        # Actor loss
        policy_distr = self.forward_actor(features[:-1])
        action_logprob = policy_distr.log_prob(actions[:-1])
        loss_policy = - action_logprob * advantage_gae.detach()

        #Entropy regularization for actor loss
        policy_entropy = policy_distr.entropy()
        loss_actor = loss_policy - self.entropy_weight * policy_entropy
        loss_actor = (loss_actor * reality_weight).mean()'''

        return loss_actor, loss_critic, policy_entropy

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict()) 
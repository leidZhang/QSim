import torch
import torch.nn as nn
import numpy as np
from .net import Block, BaseActor


class ReinFormer(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim, 
        n_blocks, 
        h_dim,  # h_dim: hidden dimension
        context_len,
        n_heads, 
        drop_p, 
        init_temperature,
        target_entropy,
        max_timestep=4096,
        dt_mask=False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len


        ### transformer blocks
        self.num_inputs = 3

        # 增加：定义卷积神经网络来处理图像
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, h_dim),
            nn.ReLU(),
        )

        input_seq_len = self.num_inputs * context_len
        blocks = [
            Block(
                h_dim, 
                input_seq_len, 
                n_heads, 
                drop_p,
                self.num_inputs,
                mgdt=False,
                dt_mask=dt_mask
            ) 
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)

        ### prediction heads
        self.predict_rtg = nn.Linear(h_dim, 1)
        # stochastic action
        self.predict_action = BaseActor(h_dim, self.act_dim)
        self.predict_state = nn.Linear(h_dim, state_dim)

        # For entropy
        self.log_temperature = torch.tensor(np.log(init_temperature))
        self.log_temperature.requires_grad = True
        self.target_entropy = target_entropy

    def temperature(self):
        return self.log_temperature.exp()


    def forward(
        self, 
        timesteps, 
        states, 
        actions, 
        returns_to_go,
        images,  # 增加：图像输入
    ):

        B, T, _ = states.shape  # B: batch size T: time steps

        time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states.float()) + time_embeddings
        action_embeddings = self.embed_action(actions.float()) + time_embeddings
        rtg_embeddings = self.embed_rtg(returns_to_go.float()) + time_embeddings

        # 增加：处理图像观测
        image_features = self.conv_net(images.view(-1, 3, 84, 84))
        image_embeddings = image_features.view(B, T, -1) + time_embeddings

        # 将图像嵌入与状态嵌入结合
        combined_embeddings = state_embeddings + image_embeddings

        # stack states, RTGs, and actions and reshape sequence as
        # (s_0, R_0, a_0, s_1, R_1, a_1, s_2, R_2, a_2 ...)
        h = (
            torch.stack(
                (
                    combined_embeddings,  # 修改：使用结合后的嵌入,
                    rtg_embeddings,
                    action_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t, R_t
        # h[:, 2, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t, R_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (s_t, R_t, a_t) in sequence.
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        rtg_preds  = self.predict_rtg(h[:, 0])            # predict rtg given s----g
        action_dist_preds = self.predict_action(h[:, 1])  # predict action given s, R----a
        state_preds = self.predict_state(h[:, 2])         # predict next state given s, R, a----s'

        return (
            rtg_preds,
            action_dist_preds, 
            state_preds, 
        )

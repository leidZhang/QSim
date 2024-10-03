import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling DQN network with LSTM for sequence processing.
    """
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()
        # CNN for image processing
        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)  # 4 stacked RGB images: 4*3=12 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer for state information
        self.fc_state = nn.Linear(6, 128)  # 6-dimensional state input

        # Combine CNN and state info
        self.fc_combined = nn.Linear(7 * 7 * 64 + 128, 512)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)

        # Advantage and Value streams
        self.fc_adv1 = nn.Linear(256, 256)
        self.fc_adv2 = nn.Linear(256, action_size)

        self.fc_val1 = nn.Linear(256, 256)
        self.fc_val2 = nn.Linear(256, 1)

    def forward(self, images, state_info, hidden):
        """
        Stacked images.
        """
        print(images)
        batch_size, seq_len, C, H, W = images.size()  # Channels, Height, Width

        # Reshape for CNN
        images = images.view(batch_size * seq_len, C, H, W)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, seq_len, -1)  # Reshape for LSTM input

        # Process state info
        state_info = state_info.view(batch_size * seq_len, -1)
        state_out = F.relu(self.fc_state(state_info))
        state_out = state_out.view(batch_size, seq_len, -1)

        # Combine CNN and state info
        x = torch.cat((x, state_out), dim=2)
        x = F.relu(self.fc_combined(x))

        # LSTM layer
        x, hidden = self.lstm(x, hidden)

        # Take the last output for dueling DQN
        x = x[:, -1, :]

        # Advantage stream
        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        # Value stream
        val = F.relu(self.fc_val1(x))
        val = self.fc_val2(val).expand(x.size(0), adv.size(1))

        # Combine streams
        q_values = val + adv - adv.mean(dim=1, keepdim=True)

        return q_values, hidden

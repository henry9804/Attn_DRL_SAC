import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(40, 8, 0.5)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(25, 300),
            nn.ReLU(),
            nn.Linear(300, 1224),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(7000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 500),
            nn.ReLU(),
            nn.Linear(500, 30),
            nn.ReLU()
        )

        self.mu = nn.Linear(30, 2)
        self.log_sigma = nn.Linear(30, 4)

    def forward(self, obs, state):
        torch.autograd.set_detect_anomaly(True)
        x = self.conv1(obs)
        x = self.cbam(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        y = self.fc1(state)
        z = torch.concat([x, y], dim=1)
        z = self.fc2(z)
        mu = self.mu(z)
        A = self.log_sigma(z)
        A = torch.reshape(A, [-1, 2, 2])
        A_t = torch.transpose(A, dim0=1, dim1=2)
        log_sigma = torch.matmul(A, A_t)
        log_sigma = torch.clamp(log_sigma, -10.0, 2.0)
        
        return mu, log_sigma

    def attn_mask(self, obs):
        x = self.conv1(obs)
        x = self.cbam(x)
        x = torch.sum(x, dim=1)

        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(40, 8, 0.5)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(25, 300),
            nn.ReLU(),
            nn.Linear(300, 1224),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(7000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 500),
            nn.ReLU(),
            nn.Linear(500, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Tanh()
        )

    def forward(self, obs, state):
        x = self.conv1(obs)
        x = self.cbam(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        y = self.fc1(state)
        z = torch.concat([x, y], dim=1)
        V = self.fc2(z)
        
        return V


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(40, 8, 0.5)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(27, 300),
            nn.ReLU(),
            nn.Linear(300, 1224),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(7000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 500),
            nn.ReLU(),
            nn.Linear(500, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Tanh()
        )

    def forward(self, obs, state, action):
        x = self.conv1(obs)
        x = self.cbam(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        y = torch.concat([state, action], dim=1)
        y = self.fc1(y)
        z = torch.concat([x, y], dim=1)
        Q = self.fc2(z)
        
        return Q


class CBAM(nn.Module):
    def __init__(self, size, channel, reduction):
        super(CBAM, self).__init__()

        self.channel_maxPool = nn.MaxPool2d(size)
        self.channel_avgPool = nn.AvgPool2d(size)

        self.channel = channel
        hidden = (int)(channel * reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel)
        )

        self.spatial_maxPool = nn.MaxPool3d([channel, 1, 1])
        self.spatial_avgPool = nn.AvgPool3d([channel, 1, 1])

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):   # x: (B, C, H, W)
        # channel attention
        channel_max = self.channel_maxPool(x)
        channel_avg = self.channel_avgPool(x)

        mlp_max = self.mlp(torch.reshape(channel_max, [-1, self.channel]))
        mlp_avg = self.mlp(torch.reshape(channel_avg, [-1, self.channel]))

        channel_attention = torch.sigmoid(mlp_max + mlp_avg)
        channel_attention = torch.reshape(channel_attention, [-1, self.channel, 1, 1])

        # channel refined feature
        x = torch.mul(x, channel_attention)

        # spatial attention
        spatial_max = self.spatial_maxPool(x)
        spatial_avg = self.spatial_avgPool(x)

        spatial = torch.concat([spatial_max, spatial_avg], dim=1)
        spatial_attention = self.conv(spatial)

        # spatial refined feature
        x = torch.mul(x, spatial_attention)

        return x


import numpy as np
import math

import torch
import torch.nn as nn
from torch.distributions import Normal

class Attn_DRL():
    def __init__(self, device='cuda'):
        super(Attn_DRL, self).__init__()

        self.policy_net = Actor().to(device)
        self.value_net = Critic().to(device)
        self.Target_value_net = Critic().to(device)
        self.Q_net1 = Q().to(device)
        self.Q_net2 = Q().to(device)

        self.device = device

    def update_target(self):
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, obs, state):
        obs = torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device)
        state = torch.FloatTensor(np.expand_dims(state, 0)).to(self.device)
        mu, log_sigma = self.policy_net(obs, state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.zeros(z.shape)
        action[:, 0] = torch.sigmoid(z[:, 0])
        action[:, 1] = torch.tanh(z[:, 1])
        action = action.detach().numpy()
        return action # return a numpy, float32

    def save(self, save_dir, ep_i, num_training):
        torch.save(self.policy_net.state_dict(), ('./'+save_dir+'/policy_net.pth'))
        torch.save(self.value_net.state_dict(), ('./'+save_dir+'/value_net.pth'))
        torch.save(self.Target_value_net.state_dict(), ('./'+save_dir+'/Target_value_net.pth'))
        torch.save(self.Q_net1.state_dict(), ('./'+save_dir+'/Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), ('./'+save_dir+'/Q_net2.pth'))
        torch.save({'num_training': num_training,
                    'num_episode': ep_i}, ('./'+save_dir+'/checkpoint.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, load_dir):
        self.policy_net.load_state_dict(torch.load(('./'+load_dir+'/policy_net.pth')))
        self.value_net.load_state_dict(torch.load( ('./'+load_dir+'/value_net.pth')))
        self.Target_value_net.load_state_dict(torch.load( ('./'+load_dir+'/Target_value_net.pth')))
        self.Q_net1.load_state_dict(torch.load(('./'+load_dir+'/Q_net1.pth')))
        self.Q_net2.load_state_dict(torch.load(('./'+load_dir+'/Q_net2.pth')))
        checkpoint = torch.load(('./'+load_dir+'/checkpoint.pth'))
        num_training = checkpoint['num_training']
        ep_i = checkpoint['num_episode']
        print("====================================")
        print("Attn_DRL model has been loaded")
        print("num training: ", num_training)
        print("num episode: ", ep_i)
        print("====================================")
        return ep_i, num_training

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
        self.log_sigma = nn.Sequential(
            nn.Linear(30, 2)
        )

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
        log_sigma = self.log_sigma(z)
        log_sigma = torch.clamp(log_sigma, -10.0, 2.0)
        
        return mu, log_sigma

    def cost_map(self, obs):
        x = self.conv1(obs)
        x = self.cbam(x)
        attn = torch.sum(x, dim=1)
        cost = attn * obs

        return cost


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
            nn.Linear(30, 1)
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
            nn.Linear(30, 1)
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
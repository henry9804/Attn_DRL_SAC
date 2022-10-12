import argparse
from collections import namedtuple
from itertools import count
import pickle
import os
from tabnanny import check
import numpy as np


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter

from my_envs.racecarGymEnv import RacecarGymEnv
from my_envs.huskyGymEnv import HuskyGymEnv
import Attn_DRL

'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()


parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)
parser.add_argument('--mode', default='train', type=str) # test or train

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=float) # discount gamma
parser.add_argument('--capacity', default=10000, type=int) # replay buffer size
parser.add_argument('--iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=128, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
args = parser.parse_args()

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

env = HuskyGymEnv(renders=args.render, isDiscrete=False)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.state.shape[0]
obs_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device)
Transition = namedtuple('Transition', ['o', 's', 'a', 'r', 'o_', 's_', 'd'])

class SAC():
    def __init__(self):
        super(SAC, self).__init__()

        self.policy_net = Attn_DRL.Actor().to(device)
        self.value_net = Attn_DRL.Critic().to(device)
        self.Target_value_net = Attn_DRL.Critic().to(device)
        self.Q_net1 = Attn_DRL.Q().to(device)
        self.Q_net2 = Attn_DRL.Q().to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        self.replay_buffer = [Transition] * args.capacity
        self.num_transition = 0 # pointer of replay buffer
        self.num_training = 0
        self.writer = SummaryWriter('./test_agent')

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def select_action(self, obs, state):
        obs = torch.FloatTensor(np.expand_dims(obs, 0)).to(device)
        state = torch.FloatTensor(np.expand_dims(state, 0)).to(device)
        mu, log_sigma = self.policy_net(obs, state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.zeros(z.shape)
        action[:, 0] = torch.sigmoid(z[:, 0])
        action[:, 1] = torch.tanh(z[:, 1])
        action = action.detach().numpy()
        return action # return a numpy, float32

    def store(self, o, s, a, r, o_, s_, d):
        index = self.num_transition % args.capacity
        transition = Transition(o, s, a, r, o_, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def evaluate(self, obs, state):
        batch_mu, batch_log_sigma = self.policy_net(obs, state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        gaussian = Normal(0, 1)

        noise = gaussian.sample()
        z = batch_mu + batch_sigma * noise.to(device)
        action = torch.zeros(z.shape).to(device)
        action[:, 0] = torch.sigmoid(z[:, 0])
        action[:, 1] = torch.tanh(z[:, 1])
        # according to original paper appendix C
        log_prob = (torch.sum(dist.log_prob(z), dim=1) - torch.log(action[:,0]*(1-action[:,0])+min_Val) - torch.log(1-action[:,1].pow(2)+min_Val)).reshape(-1, 1)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):
        torch.autograd.set_detect_anomaly(True)
        if self.num_training % 500 == 0:
            print("Training ... \t{} times ".format(self.num_training))
        o = torch.tensor(np.array([t.o for t in self.replay_buffer])).float().to(device)
        s = torch.tensor(np.array([t.s for t in self.replay_buffer])).float().to(device)
        a = torch.tensor(np.array([t.a for t in self.replay_buffer])).to(device)
        r = torch.tensor(np.array([t.r for t in self.replay_buffer])).to(device)
        o_ = torch.tensor(np.array([t.o_ for t in self.replay_buffer])).float().to(device)
        s_ = torch.tensor(np.array([t.s_ for t in self.replay_buffer])).float().to(device)
        d = torch.tensor(np.array([t.d for t in self.replay_buffer])).float().to(device)

        for _ in range(args.gradient_steps):
            #for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
            index = np.random.choice(range(args.capacity), args.batch_size, replace=False)
            bn_o = o[index].reshape(-1, 1, obs_dim[0], obs_dim[1])
            bn_s = s[index].reshape(-1, state_dim)
            bn_a = a[index].reshape(-1, action_dim)
            bn_r = r[index].reshape(-1, 1)
            bn_o_ = o_[index].reshape(-1, 1, obs_dim[0], obs_dim[1])
            bn_s_ = s_[index].reshape(-1, state_dim)
            bn_d = d[index].reshape(-1, 1)

            target_value = self.Target_value_net(bn_o_, bn_s_)
            next_q_value = bn_r + (1 - bn_d) * args.gamma * target_value

            excepted_value = self.value_net(bn_o, bn_s)
            excepted_Q1 = self.Q_net1(bn_o ,bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_o, bn_s, bn_a)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_o, bn_s)
            excepted_new_Q = torch.min(self.Q_net1(bn_o, bn_s, sample_action), self.Q_net2(bn_o, bn_s, sample_action))
            next_value = excepted_new_Q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()
            
            pi_loss = (log_prob - excepted_new_Q).mean() # according to original paper eq.12

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)

            # mini batch gradient descent

            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()
                        
            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - args.tau) + param * args.tau)
            
            self.num_training += 1

    def save(self, ep_i):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        torch.save({'num_training': self.num_training,
                    'num_episode': ep_i}, './SAC_model/checkpoint.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load( './SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        checkpoint = torch.load('./SAC_model/checkpoint.pth')
        self.num_training = checkpoint['num_training']
        ep_i = checkpoint['num_episode']
        print("model has been load")
        return ep_i


def main():
    agent = SAC()
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            obs = env.reset()
            state = env.state
            for t in count():
                action = agent.select_action(obs, state)
                next_obs, reward, done, next_state = env.step(action[0])
                ep_r += reward
                env.render()
                if done != 0:
                    if done == 1:
                        print("Trial_i \t{}, the ep_r is \t{}, the step is \t{}".format(i, ep_r, t))
                    break
                obs = next_obs
                state = next_state
    else:
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        ep_start = 0
        if(args.load):
            ep_start = agent.load()            

        for i in range(ep_start, args.iteration):
            obs = env.reset()
            state = env.state
            for t in count():
                action = agent.select_action(obs, state)
                next_obs, reward, done, next_state = env.step(action[0])
                if done == -1:
                    break
                else:
                    ep_r += reward
                    if args.render and i >= args.render_interval : env.render()
                    agent.store(obs, state, action, reward, next_obs, next_state, done)

                    if agent.num_transition >= args.capacity:
                        agent.update()

                    obs = next_obs
                    state = next_state
                    if done:
                        if i > 100:
                            print("Ep_i \t{}, the ep_r is \t{}, the step is \t{}".format(i, ep_r, t))
                        break
            if i % args.log_interval == 0:
                agent.save(i)
            agent.writer.add_scalar('ep_r', ep_r, global_step=i)
            ep_r = 0


if __name__ == '__main__':
    main()

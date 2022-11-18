import argparse
from collections import namedtuple
from itertools import count
from multiprocessing import heap
import pickle
import os
import numpy as np
import math
import random
import heapq


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter

from my_envs.huskyGymEnv import HuskyGymEnv
from DWA_RL import DWA_RL

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
parser.add_argument('--save_dir', default='DWA_RL_model', type=str)
parser.add_argument('--load_dir', default='DWA_RL_model', type=str)
parser.add_argument('--attn_dir', default='SAC_model', type=str)
args = parser.parse_args()

env = HuskyGymEnv(renders=args.render, isDiscrete=False)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
#random.seed(args.seed)

state_dim = env.state.shape[0]
obs_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
num_actions = env.config.num_v * env.config.num_w
min_Val = torch.tensor(1e-7).float().to(device)
Transition = namedtuple('Transition', ['o', 'a', 'r', 'o_', 'd'])

class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        self.dwa_rl = DWA_RL(num_actions, ('./'+args.attn_dir+'/policy_net.pth'), device)
        self.dwa_rl.update_target()

        self.optimizer = optim.Adam(self.dwa_rl.current_model.parameters(), lr=args.learning_rate)

        self.replay_buffer = [Transition] * args.capacity
        self.num_transition = 0 # pointer of replay buffer
        self.num_training = 0
        self.writer = SummaryWriter('./test_agent2')

        self.criterion = nn.MSELoss()

        os.makedirs(('./'+args.save_dir+'/'), exist_ok=True)

    def store(self, o, a, r, o_, d):
        index = self.num_transition % args.capacity
        transition = Transition(o, a, r, o_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def epsilon_by_time(self):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * self.num_training / epsilon_decay)

    def update(self):
        torch.autograd.set_detect_anomaly(True)
        if self.num_training % 500 == 0:
            print("Training ... \t{} times ".format(self.num_training))
        o = torch.tensor(np.array([t.o for t in self.replay_buffer])).float().to(device)
        a = torch.tensor(np.array([t.a for t in self.replay_buffer])).to(device)
        r = torch.tensor(np.array([t.r for t in self.replay_buffer])).to(device)
        o_ = torch.tensor(np.array([t.o_ for t in self.replay_buffer])).float().to(device)
        d = torch.tensor(np.array([t.d for t in self.replay_buffer])).float().to(device)

        for _ in range(args.gradient_steps):
            #for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
            index = np.random.choice(range(args.capacity), args.batch_size, replace=False)
            bn_o  = o[index]
            bn_a  = a[index].unsqueeze(1)
            bn_r  = r[index].unsqueeze(1)
            bn_o_ = o_[index]
            bn_d  = d[index].unsqueeze(1)

            q_values            = self.dwa_rl.current_model(bn_o)
            next_q_values       = self.dwa_rl.current_model(bn_o_)
            next_q_state_values = self.dwa_rl.target_model(bn_o_)

            q_value = q_values.gather(1, bn_a)
            next_q_value = next_q_state_values.gather(1, torch.argmax(next_q_values, 1).unsqueeze(1))
            expected_q_value = bn_r + (1 - bn_d) * args.gamma * next_q_value

            loss = self.criterion(q_value, expected_q_value.detach())
            self.writer.add_scalar('Loss/loss', loss, global_step=self.num_training)

            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.dwa_rl.current_model.parameters(), 0.5)
            self.optimizer.step()
            self.num_training += 1

        if self.num_training % 100 == 0:
            self.dwa_rl.update_target()


def main():
    agent = DQN()
    ep_r = 0
    if args.mode == 'test':
        agent.dwa_rl.load(args.load_dir)
        for i in range(args.iteration):
            obs = env.reset()
            state = env.state
            cur_pos = env.getPosition()
            cur_vel = env.getVelocity()
            x, cost_map, goal, _ = agent.dwa_rl.get_cost_goal(obs, state, cur_pos, cur_vel, env.config.scale)
            obs_space, action_space = agent.dwa_rl.dwa(x, cost_map, goal, env.config)
            for t in count():
                index = agent.dwa_rl.select_action(obs_space)
                action = action_space[index]
                next_obs, reward, done, next_state = env.step(action)
                cur_pos = env.getPosition()
                cur_vel = env.getVelocity()
                x, cost_map, goal, _ = agent.dwa_rl.get_cost_goal(next_obs, next_state, cur_pos, cur_vel, env.config.scale)
                next_obs_space, next_action_space = agent.dwa_rl.dwa(x, cost_map, goal, env.config)
                ep_r += reward
                env.render()
                if done:
                    break
                obs_space = next_obs_space
                action_space = next_action_space
    else:
        print("====================================")
        print("Collection Experience...")
        print("====================================")

        result = np.zeros(20)
        for i in range(args.iteration):
            obs = env.reset()
            state = env.state
            cur_pos = env.getPosition()
            cur_vel = env.getVelocity()
            x, cost_map, goal, _ = agent.dwa_rl.get_cost_goal(obs, state, cur_pos, cur_vel, env.config.scale)
            obs_space, action_space = agent.dwa_rl.dwa(x, cost_map, goal, env.config)
            for t in count():
                index = agent.dwa_rl.select_action(obs_space, agent.epsilon_by_time())
                action = action_space[index]
                next_obs, reward, done, next_state = env.step(action)

                if done == -1:
                    result[i%20] = 0
                    break
                else:
                    cur_pos = env.getPosition()
                    cur_vel = env.getVelocity()
                    x, cost_map, goal, _ = agent.dwa_rl.get_cost_goal(next_obs, next_state, cur_pos, cur_vel, env.config.scale)
                    next_obs_space, next_action_space = agent.dwa_rl.dwa(x, cost_map, goal, env.config)
                    agent.store(obs_space, index, reward, next_obs_space, done)

                    ep_r += reward
                    obs_space = next_obs_space
                    action_space = next_action_space

                    if args.render and i >= args.render_interval : env.render()

                    if agent.num_transition >= args.capacity:
                        agent.update()

                    if done:
                        result[i%20] = 1
                        if i > 100:
                            print("Ep_i \t{}, the ep_r is \t{}, the step is \t{}".format(i, ep_r, t))
                        break
            
            if i % args.log_interval == 0:
                agent.dwa_rl.save(args.save_dir)
            agent.writer.add_scalar('ep_r', ep_r, global_step=i)
            agent.writer.add_scalar('success_rate', np.mean(result), global_step=i)
            ep_r = 0
            if(np.mean(result)>0.9):
                agent.dwa_rl.save(args.save_dir)
                break


if __name__ == '__main__':
    main()

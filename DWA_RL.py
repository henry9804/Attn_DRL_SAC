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

from my_envs.racecarGymEnv import RacecarGymEnv
from Attn_DRL import Actor
import modules

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

env = RacecarGymEnv(renders=args.render, isDiscrete=False)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.state.shape[0]
obs_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
num_actions = env.config.num_v * env.config.num_w
min_Val = torch.tensor(1e-7).float().to(device)
Transition = namedtuple('Transition', ['o', 'a', 'r', 'o_', 'd'])

class DWA_RL():
    def __init__(self, attn_path):
        super(DWA_RL, self).__init__()

        self.attn = Actor().to(device)
        self.attn.load_state_dict(torch.load(attn_path))

        self.dwa = modules.DWA().to(device)
        self.current_model = modules.Net([4, num_actions, 1]).to(device)
        self.target_model = modules.Net([4, num_actions, 1]).to(device)

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=args.learning_rate)

        self.replay_buffer = [Transition] * args.capacity
        self.num_transition = 0 # pointer of replay buffer
        self.num_training = 0
        self.writer = SummaryWriter('./test_agent2')

        self.criterion = nn.MSELoss()

        os.makedirs('./DWA_RL_model/', exist_ok=True)
        self.update_target()


    def store(self, o, a, r, o_, d):
        index = self.num_transition % args.capacity
        transition = Transition(o, a, r, o_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def epsilon_by_time(self, time):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * time / epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

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
            bn_o = o[index].reshape(-1, 4, num_actions, 1)
            bn_a = a[index].reshape(-1, action_dim)
            bn_r = r[index].reshape(-1, 1)
            bn_o_ = o_[index].reshape(-1, 4, num_actions, 1)
            bn_d = d[index].reshape(-1, 1)

            q_values            = self.current_model(bn_o)
            next_q_values       = self.current_model(bn_o_)
            next_q_state_values = self.target_model(bn_o_)

            q_value             = q_values.gather(1, bn_a.unsqueeze(1)).squeeze(1) 
            next_q_value        = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value    = bn_r + (1 - bn_d) * args.gamma * next_q_value

            loss = self.criterion(q_value, expected_q_value).mean()
            self.writer.add_scalar('Loss/loss', loss, global_step=self.num_training)

            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.current_model.parameters(), 0.5)
            self.optimizer.step()
            self.num_training += 1

        if self.num_training % 100 == 0:
            self.update_target()


    def get_cost_goal(self, obs, state, env):
        obs = torch.tensor(np.expand_dims(obs, 0)).float().to(device)
        v, w = env._p.getBaseVelocity(env._racecar.racecarUniqueId)
        v = np.linalg.norm(v)
        w = w[2]
        x = [0, 0, 0, v, w]
        cost_map = self.attn.cost_map(obs)
        goal = self.compute_goal_position(cost_map, state, env.config.scale)
        return x, cost_map, goal

    def compute_goal_position(self, cmap, state, scale):
        cost_map = cmap.cpu().detach().numpy()
        gamma = np.pi/6
        r_explore = 0.5 + 0.00001 * (1 / np.abs(np.mean(cost_map)))
        if state[0] < r_explore:
            goal = np.array([state[0]*math.cos(state[1]), state[0]*math.sin(state[1])])
        else:
            min_cost = 10000000
            path_cost = self.dijkstra(cost_map)
            for theta in np.linspace(state[1] - gamma, state[1] + gamma, 11):
                goal_condidate = np.array([r_explore*math.cos(theta), r_explore*math.sin(theta)], dtype=int)
                if path_cost[goal_condidate[0], goal_condidate[1]] < min_cost:
                    goal = goal_condidate
                goal = goal * scale[:2]
        return goal

    def dijkstra(self, cost_map):
        # TODO
        cost_map -= np.min(cost_map)
        h, w = cost_map[0].shape
        path_cost = np.ones([h, w]) * np.inf
        start_x = (int)(h/2)
        start_y = (int)(w/2)
        path_cost[start_x, start_y] = 0
        queue = []
        heapq.heappush(queue, (path_cost[start_x, start_y], start_x, start_y))
        adjacency_list = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while(queue):
            cur_cost, x, y = heapq.heappop(queue)
            if path_cost[x, y] < cur_cost:
                continue
            for adj in adjacency_list:
                adj_x = x+adj[0]
                adj_y = y+adj[1]
                if adj_x < 0 or adj_x >= h:
                    continue
                if adj_y < 0 or adj_y >= w:
                    continue
                new_cost = cur_cost + cost_map[0, adj_x, adj_y]
                if new_cost < path_cost[adj_x, adj_y]:
                    path_cost[adj_x, adj_y] = new_cost
                    heapq.heappush(queue, (new_cost, adj_x, adj_y))
        return path_cost

    def select_action(self, obs_space, action_space):
        epsilon = self.epsilon_by_time(self.num_training)
        if random.random() > epsilon:
            q_value = self.current_model(obs_space)
            action = action_space[torch.argmax(q_value, dim=1)]
        else:
            action = action_space[random.randrange(action_space.shape[1])]
        return action # return a numpy, float32


    def save(self):
        torch.save(self.current_model.state_dict(), './DWA_RL_model/current_model.pth')
        torch.save(self.target_model.state_dict(), './DWA_RL_model/target_model.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.current_model.load_state_dict(torch.load('./DWA_RL_model/current_model.pth'))
        self.target_model.load_state_dict(torch.load('./DWA_RL_model/target_model.pth'))
        print("model has been load")


def main():
    agent = DWA_RL('./SAC_model/policy_net.pth')
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            obs = env.reset()
            state = env.state
            x, cost_map, goal = agent.get_cost_goal(obs, state, env)
            obs_space, action_space = agent.dwa(x, cost_map, goal, env.config)
            for t in count():
                action = agent.select_action(obs_space, action_space)
                next_obs, reward, done, next_state = env.step(action)
                x, cost_map, goal = agent.get_cost_goal(next_obs, next_state, env)
                next_obs_space, next_action_space = agent.dwa(x, cost_map, goal, env.config)
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

        for i in range(args.iteration):
            obs = env.reset()
            state = env.state
            x, cost_map, goal = agent.get_cost_goal(obs, state, env)
            obs_space, action_space = agent.dwa(x, cost_map, goal, env.config)
            for t in range(200):
                action = agent.select_action(obs_space, action_space)
                next_obs, reward, done, next_state = env.step(action)

                if done == -1:
                    break
                else:
                    x, cost_map, goal = agent.get_cost_goal(next_obs, next_state, env)
                    next_obs_space, next_action_space = agent.dwa(x, cost_map, goal, env.config)
                    agent.store(obs_space, action, reward, next_obs_space, done)

                    ep_r += reward
                    obs_space = next_obs_space
                    action_space = next_action_space

                    if args.render and i >= args.render_interval : env.render()

                    if agent.num_transition >= args.capacity:
                        agent.update()

                    if done:
                        if i > 100:
                            print("Ep_i \t{}, the ep_r is \t{}, the step is \t{}".format(i, ep_r, t))
                        break
            
            if i % args.log_interval == 0:
                agent.save()
            agent.writer.add_scalar('ep_r', ep_r, global_step=i)
            ep_r = 0


if __name__ == '__main__':
    main()

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import count

import torch

from Attn_DRL import Attn_DRL
from DWA_RL import DWA_RL
from my_envs.huskyGymEnv import HuskyGymEnv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--seed', default=1, type=int) # show UI or not
parser.add_argument('--attn_dir', default='SAC_model', type=str)
parser.add_argument('--dwa_dir', default='DWA_RL_model', type=str)
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
min_Val = torch.tensor(1e-7).float().to(device)
num_actions = env.config.num_v * env.config.num_w

agent1 = Attn_DRL(device)
agent2 = DWA_RL(num_actions, ('./'+args.attn_dir+'/policy_net.pth'), device)

agent1.load(args.attn_dir)
agent2.load(args.dwa_dir)

#TODO
#get optimal path
'''
waypoints = []
obs = env.reset()
while(1):
    x, cost_map, goal = agent2.get_cost_goal(obs, env.state, env)
    waypoints.append(goal)
    env._p.resetBasePositionAndOrientation(env._husky.robotId, [*goal, 0.0])
    obs, done = env.getObservation()
    if done:
        break

print(waypoints)
'''

ep_r = 0
obs = env.reset()
state = env.state
traj1 = env.getPosition()
for t in count():
    action = agent1.select_action(obs, state)
    next_obs, reward, done, next_state = env.step(action[0])
    ep_r += reward
    env.render()
    traj1 = np.vstack((traj1, env.getPosition()))
    if done != 0:
        if done == 1:
            print("Attn_DRL only, the ep_r is \t{}, the step is \t{}".format(ep_r, t))
        break
    obs = next_obs
    state = next_state

ep_r = 0
obs = env.reset()
state = env.state
traj2 = env.getPosition()
v, w = env.getVelocity()
x, cost_map, goal = agent2.get_cost_goal(obs, state, [v, w], env.config.scale)
obs_space, action_space = agent2.dwa(x, cost_map, goal, env.config)
for t in count():
    index = agent2.select_action(obs_space)
    action = action_space[index]
    next_obs, reward, done, next_state = env.step(action)
    v, w = env.getVelocity()
    x, cost_map, goal = agent2.get_cost_goal(next_obs, next_state, [v, w], env.config.scale)
    next_obs_space, next_action_space = agent2.dwa(x, cost_map, goal, env.config)
    ep_r += reward
    env.render()
    traj2 = np.vstack((traj2, env.getPosition()))
    if done != 0:
        if done == 1:
            print("Attn_DRL + DWA_RL, the ep_r is \t{}, the step is \t{}".format(ep_r, t))
        break
    obs_space = next_obs_space
    action_space = next_action_space

goal, _ = env._p.getBasePositionAndOrientation(env._cubeUniqueId)
plt.plot(goal[0], goal[1], 'gx', label='goal point')
plt.plot(traj1[:,0], traj1[:,1], 'b', label='Attn_DRL only')
plt.plot(traj2[:,0], traj2[:,1], 'r', label='Attn_DRL + DWA_RL')
plt.legend()
plt.show()
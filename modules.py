import numpy as np
import math

import torch
import torch.nn as nn

class DWA(nn.Module):
    def __init__(self):
        super(DWA, self).__init__()

    # Calculate trajectory, costings, and sort velocities according to costings
    def forward(self, x, mask, goal, config):
        dw = self.calc_dynamic_window(x, config)
        
        xinit = x[:]
        matrices = []
        # evaluate all trajectory with sampled input in dynamic window
        for v in np.linspace(dw[0], dw[1], config.num_v):
            for w in np.linspace(dw[2], dw[3], config.num_w):
                traj = self.calc_trajectory(xinit, v, w, config)
                # calc costs with weighted gains
                # speed_cost = (config.max_speed - traj[-1, 3]) * config.speed_cost_gain
                to_goal_cost = self.calc_to_goal_cost(traj, goal) * config.to_goal_cost_gain
                el_cost = self.calc_elevation_cost(traj, mask, config.scale) * config.ele_cost_gain            
                final_cost = to_goal_cost + el_cost
                
                matrices.append((v, w, el_cost, to_goal_cost, final_cost))

        dtype = [('v', float), ('w', float), ('el_cost', float), ('goal_cost', float), ('total_cost', float)]
        matrices = np.array(matrices, dtype=dtype)
        matrices = np.sort(matrices, order='total_cost')

        obs_space = np.array([matrices['v'], matrices['w'], matrices['el_cost'], matrices['goal_cost']]).reshape([4, -1, 1])
        action_space = np.array([matrices['v'], matrices['w']]).transpose()

        return obs_space, action_space

    # Model to determine the expected position of the robot after moving along trajectory
    def motion(self, x, u, dt):
        # motion model
        # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        
        x[3] = u[0]
        x[4] = u[1]

        return x

    # Calculate a trajectory sampled across a prediction time
    def calc_trajectory(self, xinit, v, w, config):

        x = np.array(xinit)
        traj = np.array(x)  # many motion models stored per trajectory
        time = 0
        while time <= config.predict_time:
            # store each motion model along a trajectory
            x = self.motion(x, [v, w], config.dt)
            traj = np.vstack((traj, x))
            time += config.dt # next sample

        return traj

    def calc_elevation_cost(self, traj, mask, scale):
        cost = 0
        (h, w) = mask[0].shape
        for x in traj:
            row = (int)(h/2 + x[1]/scale[1])
            col = (int)(w/2 - x[0]/scale[0])
            cost += mask[0, row, col]
        return cost

    # Calculate goal cost via Pythagorean distance to robot
    def calc_to_goal_cost(self, traj, goal):
        # If-Statements to determine negative vs positive goal/trajectory position
        # traj[-1,0] is the last predicted X coord position on the trajectory
        if (goal[0] >= 0 and traj[-1,0] < 0):
            dx = goal[0] - traj[-1,0]
        elif (goal[0] < 0 and traj[-1,0] >= 0):
            dx = traj[-1,0] - goal[0]
        else:
            dx = abs(goal[0] - traj[-1,0])
        # traj[-1,1] is the last predicted Y coord position on the trajectory
        if (goal[1] >= 0 and traj[-1,1] < 0):
            dy = goal[1] - traj[-1,1]
        elif (goal[1] < 0 and traj[-1,1] >= 0):
            dy = traj[-1,1] - goal[1]
        else:
            dy = abs(goal[1] - traj[-1,1])

        cost = math.sqrt(dx**2 + dy**2)
        return cost

    # Determine the dynamic window from robot configurations
    def calc_dynamic_window(self, x, config):

        # Dynamic window from robot specification
        Vs = [config.min_speed, config.max_speed,
            -config.max_yawrate, config.max_yawrate]

        # Dynamic window from motion model
        Vd = [x[2] - config.max_accel * config.dt,
            x[2] + config.max_accel * config.dt,
            x[3] - config.max_dyawrate * config.dt,
            x[3] + config.max_dyawrate * config.dt]

        #  [vmin, vmax, yawrate min, yawrate max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 14, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_size[0]),
            nn.ReLU()
        )

    def fc_input_size(self):
        x = self.conv(torch.zeros(1, *self.input_size))
        x = torch.reshape(x, [1, -1])
        return x.shape[1]

    def forward(self, obs):
        x = self.conv(obs)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
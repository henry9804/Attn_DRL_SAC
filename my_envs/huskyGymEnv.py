import os, inspect
from warnings import catch_warnings
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import cv2
import matplotlib.pyplot as plt
from . import husky
import random
from pybullet_utils import bullet_client as bc
import my_envs.pybullet_data as pd
from my_envs.heightfield import create_field
from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class Config():
    # simulation parameters

    def __init__(self):
        # robot parameter
        #NOTE good params:
        #NOTE 0.55,0.1,1.0,1.6,3.2,0.15,0.05,0.1,1.7,2.4,0.1,3.2,0.18
        self.max_speed = 0.65  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yawrate = 1.0  # [rad/s]
        self.max_accel = 2.5  # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        self.num_v = 10  # [num]
        self.num_w = 10  # [num]
        self.dt = 0.5  # [s]
        self.predict_time = 1.5  # [s]
        self.to_goal_cost_gain = 2.4 #lower = detour
        self.speed_cost_gain = 0.1 #lower = faster
        self.ele_cost_gain = 3.2 #lower z= fearless
        self.robot_radius = 0.15  # [m]
        self.scale = np.array([.1, .1, .1])
        self.obsDim = np.array([40, 40])

class HuskyGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pd.getDataPath(),
               actionRepeat=50,
               timeOut=200,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=False,
               config=Config()):
    print("init")
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._cubeUniqueId = -1
    self._envStepCounter = 0
    self._timeOut = timeOut
    self._renders = renders
    self._isDiscrete = isDiscrete
    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._p = bc.BulletClient()

    self.seed()
    #self.reset()
    self.config = config
    #self.height_map = create_field(0, meshScale=self.config.scale)

    self.prev_state = np.zeros(25, dtype=np.float32)
    self.state = np.zeros(25, dtype=np.float32)
    observation_high = np.ones(config.obsDim, dtype=np.float32) * 1000  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim, dtype=np.float32)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)
    self.viewer = None

  def reset(self):
    self._p.resetSimulation()
    self._p.setTimeStep(self._timeStep)

    #create_field(1, meshScale=self.config.scale, heightMap=self.height_map)
    self.height_map = create_field(0, meshScale=self.config.scale)
    bodyPath = os.path.join(self._urdfRoot, 'husky/husky.urdf')
    self._husky = husky.Husky(body_path=bodyPath, init_pos=[0, 0, 0.1])

    self.dist = 5 + 2. * random.random()
    ang = 3.1415925438 * random.random() - 1.5707962719

    cubex = self.dist * math.sin(ang)
    cubey = self.dist * math.cos(ang)
    cubez = 1

    self._cubeUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot, "cube.urdf"),
                                          [cubex, cubey, cubez],
                                          useFixedBase = 1)
    self._p.setGravity(0, 0, -10)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    obs, _ = self.getObservation()
    return obs

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getObservation(self):
    carpos, carorn = self._p.getBasePositionAndOrientation(self._husky.robotId)
    targetpos, targetorn = self._p.getBasePositionAndOrientation(self._cubeUniqueId)
    invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
    targetPosInCar, _ = self._p.multiplyTransforms(invCarPos, invCarOrn, targetpos, targetorn)

    # state
    carorn = pybullet.getEulerFromQuaternion(carorn)
    self.state[0] = np.linalg.norm(targetPosInCar)                    # d_goal
    self.state[1] = math.atan2(targetPosInCar[1], targetPosInCar[0])  # a_goal
    self.state[2]  = math.acos(np.dot(carpos, targetpos)/(np.linalg.norm(carpos)*np.linalg.norm(targetpos)))  # a_rel
    self.state[3]   = carorn[0] # roll
    self.state[4]  = carorn[1]  # pitch
    
    (map_h, map_w) = self.height_map.shape
    carPosInImg_row = (int)(map_h/2 + carpos[1]/self.config.scale[1])
    carPosInImg_col = (int)(map_w/2 - carpos[0]/self.config.scale[0])
    if carPosInImg_row<0 or carPosInImg_row>=map_h or carPosInImg_col<0 or carPosInImg_col>=map_w:
      # car is out of world
      observation = None
      done = -1
    else:
      # get observation
      angle = carorn[2] - np.pi/2
      if angle > np.pi:
        angle = 2*np.pi - angle
      angle = angle * (-180/np.pi)
      matrix = cv2.getRotationMatrix2D(center=(carPosInImg_col, carPosInImg_row), angle=-angle, scale=1)
      h = (int)(self.config.obsDim[0]/2)
      w = (int)(self.config.obsDim[1]/2)
      image = cv2.warpAffine(src=self.height_map, M=matrix, dsize=(map_w, map_h))
      image = cv2.copyMakeBorder(image, h, h, w, w, cv2.BORDER_CONSTANT, value=255)
      observation = image[carPosInImg_row:carPosInImg_row+self.config.obsDim[0], carPosInImg_col:carPosInImg_col+self.config.obsDim[1]] * self.config.scale[2]
      
      grad = np.gradient(observation)
      self.state[5:] = grad[0][0:20, 20]
      
      # normalize observation
      c = 0.1
      e_max = np.max(observation)
      e_min = np.min(observation)
      observation -= c
      observation += 0.1 * (e_max-e_min) * (observation>0)
      
      # done check
      done = 0
      if self.state[0] < 0.3:
        done = 1
      elif self._envStepCounter > self._actionRepeat * self._timeOut:
        done = -1

    return observation, done

  def getVelocity(self):
    v, w = self._p.getBaseVelocity(self._husky.robotId)
    v = np.linalg.norm(v)
    w = w[2]
    return v, w

  def getPosition(self):
    pos, _ = self._p.getBasePositionAndOrientation(self._husky.robotId)
    return np.array(pos)

  def step(self, action):
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._husky.robotId)
      #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)

    if (self._isDiscrete):
      lin = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
      ang = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
      linVel = lin[action]
      angVel = ang[action]
      realaction = [linVel, angVel]
    else:
      realaction = action

    self.prev_state = np.copy(self.state)
    self._husky.applyAction(realaction)
    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      obs, done = self.getObservation()
      if done != 0:
        break
      self._envStepCounter += 1
    reward = self._reward()
    
    if done == 1:
      reward += (np.float32)(1000.0)
    elif done == -1:
      reward -= (np.float32)(1000.0)
    
    reward = reward * (np.float32)(0.1)
    
    return obs, reward, done, self.state

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos, orn = self._p.getBasePositionAndOrientation(self._husky.robotId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _reward(self):
    w = 0.9 ** np.arange(0, 20, 1, dtype=np.float32)
    r_dist   = -(self.state[0] / self.dist)
    r_head   = -(np.abs(self.state[1]) / np.pi)
    r_stable = (math.cos(self.state[3])**2 + math.cos(self.state[4])**2) / 2
    r_grad   = -np.abs(np.dot(w, self.state[5:]))
    beta = np.array([1.2, 1.0, 0.01, 5.0], dtype=np.float32)
    reward = np.dot(beta, np.array([r_dist, r_head, r_stable, r_grad], dtype=np.float32))
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

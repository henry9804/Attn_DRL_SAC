import os, inspect
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
from . import racecar
import random
from pybullet_utils import bullet_client as bc
import my_envs.pybullet_data as pd
from my_envs.heightfield import create_field
from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class RacecarGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pd.getDataPath(),
               actionRepeat=50,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=False):
    print("init")
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._cubeUniqueId = -1
    self._envStepCounter = 0
    self._renders = renders
    self._isDiscrete = isDiscrete
    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._p = bc.BulletClient()

    self.seed()
    #self.reset()
    self.state = np.zeros(25, dtype=np.float32)    
    self.obsDim = np.array([40, 40])
    observation_high = np.ones(self.obsDim) * 1000  #np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(9)
    else:
      action_dim = 2
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)
    self.viewer = None

  def reset(self):
    self._p.resetSimulation()
    self._p.setTimeStep(self._timeStep)

    self.scale = [.2, .2, .1]
    self.height_map = create_field(1, meshScale=self.scale)
    self._racecar = racecar.Racecar(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep, scale=2.0)

    dist = 5 + 2. * random.random()
    ang = 2. * 3.1415925438 * random.random()

    cubex = dist * math.sin(ang)
    cubey = dist * math.cos(ang)
    cubez = 1

    self._cubeUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot, "cube.urdf"),
                                          [cubex, cubey, cubez],
                                          useFixedBase = 1)
    self._p.setGravity(0, 0, -10)
    self._envStepCounter = 0
    for i in range(100):
      self._p.stepSimulation()
    obs = self.getObservation()
    return obs

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def quat2euler(self, quat):    
    t0 = 2.0 * (quat[3] * quat[0] + quat[1] * quat[2])
    t1 = 1.0 - 2.0 * (quat[0]**2 + quat[1]**2)
    roll_x = math.atan2(t0, t1)
    t2 = 2.0 * (quat[3] * quat[1] - quat[2] * quat[0])
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = math.asin(t2)
    t3 = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    t4 = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
    yaw_z = math.atan2(t3, t4)
    return (roll_x, pitch_y, yaw_z)

  def getObservation(self):
    carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    targetpos, targetorn = self._p.getBasePositionAndOrientation(self._cubeUniqueId)
    invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
    targetPosInCar, _ = self._p.multiplyTransforms(invCarPos, invCarOrn, targetpos, targetorn)

    # state
    carorn = self.quat2euler(carorn)
    self.state[0] = np.linalg.norm(targetPosInCar)                    # d_goal
    self.state[1] = math.atan2(targetPosInCar[1], targetPosInCar[0])  # a_goal
    self.state[2]  = math.acos(np.dot(carpos, targetpos)/(np.linalg.norm(carpos)*np.linalg.norm(targetpos)))  # a_rel
    self.state[3]   = carorn[0] # roll
    self.state[4]  = carorn[1]  # pitch
    
    # observation
    (map_h, map_w) = self.height_map.shape
    carPosInImg_row = (int)(map_h/2 + carpos[1]/self.scale[1])
    carPosInImg_col = (int)(map_w/2 - carpos[0]/self.scale[0])
    angle = carorn[2] - np.pi/2
    if angle > np.pi:
      angle = 2*np.pi - angle
    angle = angle * (-180/np.pi)
    matrix = cv2.getRotationMatrix2D(center=(carPosInImg_col, carPosInImg_row), angle=-angle, scale=1)
    image = cv2.warpAffine(src=self.height_map, M=matrix, dsize=(map_w, map_h))
    r = carPosInImg_row - (int)(self.obsDim[0]/2)
    c = carPosInImg_col - (int)(self.obsDim[1]/2)
    observation = image[r:r+self.obsDim[0], c:c+self.obsDim[1]] * self.scale[2]
    
    grad = np.gradient(observation)
    self.state[5:] = grad[0][0:20, 20]
    
    # normalize observation
    c = 0.1
    e_max = np.max(observation)
    e_min = np.min(observation)
    observation -= c
    observation += 0.1 * (e_max-e_min) * (observation>0)
    
    return observation

  def step(self, action):
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
      #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)

    if (self._isDiscrete):
      fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
      steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
      forward = fwd[action]
      steer = steerings[action]
      realaction = [forward, steer]
    else:
      realaction = action

    self._racecar.applyAction(realaction)
    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      obs = self.getObservation()

      if self._termination():
        break
      self._envStepCounter += 1
    reward = self._reward()
    done = self._termination()

    return obs, reward, done, self.state

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
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

  def _termination(self):
    done = 1 if self.state[0] < 2 else 0
    return done

  def _reward(self):
    w = 0.9 ** np.arange(0, 20, 1, dtype=np.float32)
    r_dist   = -self.state[0]
    r_head   = -np.abs(self.state[1])
    r_stable = math.cos(self.state[3])**2 + math.cos(self.state[4])**2
    r_grad   = -np.dot(w, self.state[5:])
    beta = np.array([0.5, 1.0, 1.0, 0.1], dtype=np.float32)
    reward = np.dot(beta, np.array([r_dist, r_head, r_stable, r_grad], dtype=np.float32))
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

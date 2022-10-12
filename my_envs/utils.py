import numpy as np
import math
import pybullet

def vec2quat(vec, u3):
  yaw = math.atan2(vec[1], vec[0])
  pitch = math.atan2(vec[2], np.linalg.norm(vec[:2]))
  roll = 0.0
  return pybullet.getQuaternionFromEuler([roll, pitch, yaw])
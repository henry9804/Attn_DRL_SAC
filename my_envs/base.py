import os
import numpy as np
import pybullet as p
import pybullet_data

from .utils import vec2quat

class baseBody(object):
    def __init__(self, body_path, init_pos=[0,0,0], init_orient=None):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if init_orient is not None:
            self.robotId = p.loadURDF(body_path, init_pos, init_orient)
        else:
            self.robotId = p.loadURDF(body_path, init_pos)

    def getDistance(self, pos):
        Robpos, Robori = p.getBasePositionAndOrientation(self.robotId)
        distance = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([Robpos[0], Robpos[1]]))
        return distance

    @staticmethod
    def getPosDirectionfromPath(path, z=0.):
        dx = path[1, 0] - path[0, 0]
        dy = path[1, 1] - path[0, 1]
        dz = path[1, 2] - path[0, 2]
        direction = np.array([dx, dy, dz])
        direction /= np.linalg.norm(direction)
        pos = [path[0,0], path[0,1], path[0,2] + z]
        return pos, direction

    def resetBody(self, pos, direction, normal_vector=[0,0,1]):
        init_ori = vec2quat(direction, u3=normal_vector)
        p.resetBasePositionAndOrientation(self.robotId, pos, init_ori)

    def getRealVel(self, robAng):
        linVel, angVel = p.getBaseVelocity(self.robotId)
        yaw = robAng[2]
        headVec = np.array([np.cos(yaw), np.sin(yaw)])
        vel = np.array([linVel[0], linVel[1]])
        velDirection = 1. if np.dot(headVec, vel) >= 0 else -1.
        ReallinVel = velDirection * np.linalg.norm(vel)
        return ReallinVel, angVel[2]

    def getState(self):
        robPos, robOrn = p.getBasePositionAndOrientation(self.robotId)
        robAng = p.getEulerFromQuaternion(robOrn)
        linVel, angVel = p.getBaseVelocity(self.robotId)
        return robPos, robAng, linVel, angVel

    def setmaxForce(self, maxForce):
        self.maxForce = maxForce
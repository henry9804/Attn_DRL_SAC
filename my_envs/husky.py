import numpy as np
import pybullet as p

from .base import baseBody

class Husky(baseBody):
    def __init__(self, body_path='husky/husky.urdf',
                 init_pos=[0,0,0], init_orient=None):
        super().__init__(body_path, init_pos, init_orient)
        self.com = p.getLinkState(self.robotId, 0)[2]
        self.width = 0.5708
        self.wheelRadius = 0.17775
        self.maxWheelVel = 10.0
        self.wheels = [0, 1, 2, 3]
        self.maxForce = 100
        # p.changeDynamics(self.robotId, linkIndex=-1, mass=self.mass, collisionMargin=self.collisionMargin)

    def ackermannControl(self, acc, steer, dt):
        tarlinVel, tarangVel = self.ackerman2targetVel(acc, steer, dt=dt)
        wheelLeftVel, wheelRightVel = self.targetVel2wheelVel(tarlinVel, tarangVel)
        self.control(wheelLeftVel, wheelRightVel)
        
    def applyAction(self, action):
        wheelLeftVel, wheelRightVel = self.targetVel2wheelVel(action[0], action[1])
        self.control(wheelLeftVel, wheelRightVel)

    def ackerman2targetVel(self, acc, steer, dt):
        """
        s + v + -> w + r -
        s + v - -> w - r -
        s - v + -> w - r +
        s - v - -> w + r +
        """
        robPos, robOrn = p.getBasePositionAndOrientation(self.robotId)
        robAng = p.getEulerFromQuaternion(robOrn)
        currVel, currAngVel = self.getRealVel(robAng)
        TargetVel = currVel + acc * dt
        TargetAngVel = currVel * np.tan(steer) / self.wheelBase
        TargetAngVel = max(min(TargetAngVel, self.maxAngVel), -self.maxAngVel)

        return TargetVel, TargetAngVel

    def targetVel2wheelVel(self, targetlinVel, targetangVel):
        # https://en.wikipedia.org/wiki/Differential_wheeled_robot#Kinematics_of_Differential_Drive_Robots
        wheelLeftVel = (targetlinVel - targetangVel * self.width / 2.0) / self.wheelRadius
        wheelRightVel = (targetlinVel + targetangVel * self.width / 2.0) / self.wheelRadius
        wheelLeftVel = max(min(wheelLeftVel, self.maxWheelVel), -self.maxWheelVel)
        wheelRightVel = max(min(wheelRightVel, self.maxWheelVel), -self.maxWheelVel)
        return wheelLeftVel, wheelRightVel

    def control(self, wheelLeftVel, wheelRightVel):
        wheelVelocities =  [wheelLeftVel, wheelRightVel, wheelLeftVel, wheelRightVel]

        for i in range(len(self.wheels)):
            p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                    jointIndex=self.wheels[i],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=wheelVelocities[i],
                                    force=self.maxForce)

    def keycontrol(self, keys):
        wheelVelocities = [0, 0, 0, 0]
        LatTargetSpeed = 5
        LongTargetSpeed = 5

        if p.B3G_LEFT_ARROW in keys:
            for i in range(len(self.wheels)):
                wheelVelocities[i] = wheelVelocities[i] - LatTargetSpeed * self.wheelDeltasTurn[i]
        if p.B3G_RIGHT_ARROW in keys:
            for i in range(len(self.wheels)):
                wheelVelocities[i] = wheelVelocities[i] + LatTargetSpeed * self.wheelDeltasTurn[i]
        if p.B3G_UP_ARROW in keys:
            for i in range(len(self.wheels)):
                wheelVelocities[i] = wheelVelocities[i] + LongTargetSpeed * self.wheelDeltasFwd[i]
        if p.B3G_DOWN_ARROW in keys:
            for i in range(len(self.wheels)):
                wheelVelocities[i] = wheelVelocities[i] - LongTargetSpeed * self.wheelDeltasFwd[i]

        for i in range(len(self.wheels)):
            p.setJointMotorControl2(self.robotId,
                                    self.wheels[i],
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=wheelVelocities[i],
                                    force=self.maxForce)

        print(f'\tKeywheelLeftVel : {wheelVelocities[0]:.2f}, KeywheelRightVel : {wheelVelocities[1]:.2f}')
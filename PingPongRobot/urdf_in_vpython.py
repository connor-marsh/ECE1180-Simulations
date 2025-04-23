print("HIIIIIIIIIIIIIIII")
# from vpython import *
from urdfpy import URDF
robot = URDF.load("PingPongRobot/arm_urdf.urdf").robot
for link in robot.links:
    print("AAAAAAAAA")
    print(link.name)
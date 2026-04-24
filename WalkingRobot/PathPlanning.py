from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from WalkingRobot import WalkingRobot

# LOAD MAP + PRM
data = rtb_load_matfile("data/house.mat")
floorplan = data["floorplan"]
places = data["places"]

prm = PRMPlanner(occgrid=floorplan, seed=0)
prm.plan(npoints=300)

rooms = list(places.keys())
rd.shuffle(rooms)

start = places[rooms[0]]
goal = places[rooms[1]]

path = np.array(prm.query(start, goal))

# MAP FIGURE (create FIRST)
fig_map, ax_map = plt.subplots()
ax_map.imshow(floorplan, cmap="Reds", origin='lower')
ax_map.plot(path[:,0], path[:,1], 'b', linewidth=2)
ax_map.plot(start[0], start[1], 'go')
ax_map.plot(goal[0], goal[1], 'ro')
ax_map.set_title("House Map + PRM Path")
ax_map.set_aspect('equal')

plt.show(block=False) 


# ROBOT 
robot = WalkingRobot()
robot.followPath(path / 100)

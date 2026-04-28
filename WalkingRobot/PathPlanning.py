from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from WalkingRobot import WalkingRobot

rd.seed(80085)

def DrawRobotOnMap(x, y, angle):
    # convert meters → cm (map is in cm)
    x_cm = x * 100
    y_cm = y * 100

    # update dot position
    robot_dot.set_data([x_cm], [y_cm])

    # create the arrow
    dx = np.cos(angle)
    dy = np.sin(angle)

    robot_arrow.set_offsets([x_cm, y_cm])
    robot_arrow.set_UVC(dx, dy)



# LOAD MAP + PRM
data = rtb_load_matfile("data/house.mat")
floorplan = data["floorplan"]
places = data["places"]

# Create a floor plan
prm = PRMPlanner(occgrid=floorplan, seed=0)
prm.plan(npoints=300)
current_path_line = None

# Create a list of rooms in random order
rooms = list(places.keys())
rd.shuffle(rooms)

# Map figure
fig_map, ax_map = plt.subplots()
ax_map.imshow(floorplan, cmap="Reds", origin='lower')
ax_map.set_aspect('equal')

# Make sure to fit the entire house
ax_map.set_xlim(0, floorplan.shape[1])
ax_map.set_ylim(0, floorplan.shape[0])

# Robot position and direction arrow
robot_dot, = ax_map.plot([], [], color="g", markersize=4, zorder=10)
robot_arrow = ax_map.quiver([], [], [], [], color='r', scale=20, zorder=5)

# Start and end position
current_start_dot = None
current_goal_dot = None

plt.show(block=False)

# create robot
robot = WalkingRobot()
robot.debug_callback = DrawRobotOnMap


# Create random paths. We interpreted a nested path A->B->C->D... -> Z, as a logical solution,
# but the program can be modified to pick random start and goal for every iteration.
for i in range(5):
    # Path 1: A -> B, 
    # Path 2: B -> C, 
    # Path 3: C -> D ... 
    start = places[rooms[i]]
    goal = places[rooms[i+1]]

    # Calculate path
    path = np.array(prm.query(start, goal))

    # Remove previous path
    if current_path_line is not None:
        current_path_line.remove()
        current_start_dot.remove()
        current_goal_dot.remove()


    # Plot start and goal points
    current_start_dot, = ax_map.plot(start[0], start[1], marker='o', color='green', markersize=8, zorder=15)
    current_goal_dot, = ax_map.plot(goal[0], goal[1], marker='o', color='red', markersize=8, zorder=15)

        
    # Plot abd store path
    current_path_line, = ax_map.plot(path[:,0], path[:,1], linewidth=2, color="b")

    # Follow path (converted from cm -> m)
    robot.followPath(path / 100)

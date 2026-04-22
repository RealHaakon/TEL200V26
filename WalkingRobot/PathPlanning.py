from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Load file and get floorplan
data = rtb_load_matfile("data/house.mat")
floorplan = data["floorplan"]

places = data["places"]

# prm planner
prm = PRMPlanner(occgrid=floorplan, seed=0);
prm.plan(npoints=300)
for i in range(4):

    # Select random rooms
    rooms = list(places.keys())
    rd.shuffle(rooms)

    start = places[rooms[0]]
    goal = places[rooms[1]]

    # Create path between rooms
    path = prm.query(start, goal)
    path = np.array(path)

    print(path)

    # Display house
    plt.figure()
    plt.imshow(floorplan, cmap="Reds", origin='lower')

    plt.title("House Floorplan")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")

    plt.gca().set_aspect('equal')


    plt.plot(path[:,0], path[:,1], 'b', linewidth=3)
    plt.plot(start[0], start[1], 'go', markersize=10)
    plt.plot(goal[0], goal[1], 'bo', markersize=10)


    plt.show()
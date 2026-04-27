from roboticstoolbox import *
import matplotlib.pyplot as plt

numNodes = 300

# load map
data = rtb_load_matfile("data/house.mat")
floorplan = data["floorplan"]

# Create a PRM map
prm = PRMPlanner(occgrid=floorplan, seed=0)
prm.plan(npoints=numNodes)
prm.plot()

plt.title(f"PRM map with {numNodes} nodes")
plt.savefig(f"PRM{numNodes}")

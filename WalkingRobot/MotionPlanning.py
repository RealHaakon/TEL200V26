from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialgeometry import Cuboid
import math

pi = math.pi

# Copyright (C) 1993-2021, by Peter I. Corke
mm = 0.001
L1 = 100 * mm
L2 = 100 * mm

print('create leg model\n')

# now create a robot to represent a single leg
leg = ERobot(ET.Rz() * ET.Rx() * ET.ty(L1) * ET.Rx() * ET.tz(-L2))

# define the key parameters of the gait trajectory, 
# walking in the x-direction

# forward and backward limits for foot on ground
x_forward = 50; x_backward = -x_forward; 

# height of foot when up and down
z_up = -20; z_down = -50;     

# distance of foot from body along y-axis
y_dist = -50;              

# define the rectangular path taken by the foot
segments = np.array([
    [x_forward, y_dist, z_down],
    [x_backward, y_dist, z_down],
    [x_backward, y_dist, z_up],
    [x_forward, y_dist, z_up],
    [x_forward, y_dist, z_down]
]) * mm

# build the gait. the points are:
#   1 start of walking stroke
#   2 end of walking stroke
#   3 end of foot raise
#   4 foot raised and forward
#
# The segments times are :
#   1->2  3s
#   2->3  0.5s
#   3->4  1s
#   4->1  0.5ss
#
# A total of 4s, of which 3s is walking and 1s is reset.  At 0.01s sample
# time this is exactly 400 steps long.
#
# We use a finite acceleration time to get a nice smooth path, which means
# that the foot never actually goes through any of these points.  This
# makes setting the initial robot pose and velocity difficult.
#
# Intead we create a longer cyclic path: 1, 2, 3, 4, 1, 2, 3, 4. The
# first 1->2 segment includes the initial ramp up, and the final 3->4
# has the slow down.  However the middle 2->3->4->1 is smooth cyclic
# motion so we "cut it out" and use it.
print('create trajectory\n')
traj = mstraj(segments, tsegment=[3, 0.25, 0.5, 0.25], dt=0.01, tacc=0.2)
print('inverse kinematics (this will take a moment)....', end='')

xcycle = traj.q
xcycle = np.vstack((xcycle, xcycle[-3:,:]))

sol = leg.ikine_LM( SE3(xcycle), mask=[1, 1, 1, 0, 0, 0] )

print(' done')

qcycle = sol.q
print(xcycle.shape)

# dimensions of the robot's rectangular body, width and height, the legs
# are at each corner.
W = 100 * mm; L = 200 * mm


# create 4 leg robots.  Each is a clone of the leg robot we built above,
# has a unique name, and a base transform to represent it's position
# on the body of the walking robot.
legs = [
    ERobot(leg, name='leg0'),
    ERobot(leg, name='leg1'),
    ERobot(leg, name='leg2'),
    ERobot(leg, name='leg3')
]

from roboticstoolbox.backends.PyPlot import PyPlot

# Create an enviroment for the robot
env = PyPlot()
scale = 200 * mm

env.launch(limits=[-scale, scale, -scale, scale, -0.15, 0.05])

# Rotational matrix for adjusting the legs

# instantiate each robot in the backend environment
for leg in legs:
    leg.q = np.r_[0, 0, 0]
    env.add(leg, readonly=True, jointaxes=False, eeframe=False, shadow=False)

# Create the robot
body = Cuboid([L, W, 30 * mm], color='b')
body.base = SE3(0, 0, 0)
T = body.base
env.add(body)

# Rotation for legs
leg_adjustment = SE3.Rz(pi)

# Update leg positions
legs[0].base = T * SE3( L / 2, -W / 2, 0) 
legs[1].base = T * SE3(-L / 2, -W / 2, 0) 
legs[2].base = T * SE3( L / 2,  W / 2, 0) * leg_adjustment
legs[3].base = T * SE3(-L / 2,  W / 2, 0) * leg_adjustment


env.step()

def gait(cycle, k, offset, flip):
    k = (k + offset) % cycle.shape[0]
    q = cycle[k, :].copy()
    if flip:
        q[0] = -q[0]   # for left-side legs
    return q

env.step()


# Robot pose
x = 0
y = 0
angle = 0


def walk100mm(x, y, angle):
    # movement length per qcycle
    walking_speed = 100*mm 

    for i in range(400):
        # Check if quit application
        if not plt.fignum_exists(env.fig.number):
            break
        
        x += walking_speed / 400 * math.cos(angle) 
        y += walking_speed / 400 * math.sin(angle)

        # update robot position
        T = SE3(x, y, 0) * SE3.Rz(angle)
        body.base = T

        # Update leg positions
        legs[0].base = T * SE3( L / 2, -W / 2, 0) 
        legs[1].base = T * SE3(-L / 2, -W / 2, 0) 
        legs[2].base = T * SE3( L / 2,  W / 2, 0) * leg_adjustment
        legs[3].base = T * SE3(-L / 2,  W / 2, 0) * leg_adjustment


        legs[0].q = gait(qcycle, i, 0, False)
        legs[1].q = gait(qcycle, i, 100, False)
        legs[2].q = gait(qcycle, i, 200, True)
        legs[3].q = gait(qcycle, i, 300, True)
        env.step(dt=0.02)

    return x, y, angle

def turn1deg(x, y, angle, clockwise = False):
    turning_speed = 1 * 2 * pi / 360 # rotation angle per qcycle

    if (clockwise): 
        turning_speed *= -1
    
    for i in range(400):
        # Check if quit application
        if not plt.fignum_exists(env.fig.number):
            break

        angle += turning_speed / 400
        
        # update robot position
        T = SE3(x, y, 0) * SE3.Rz(angle)
        body.base = T

        # Update leg positions
        legs[0].base = T * SE3( L / 2, -W / 2, 0)                  # front  left   
        legs[1].base = T * SE3(-L / 2, -W / 2, 0)                  # back   left  
        legs[2].base = T * SE3( L / 2,  W / 2, 0) * leg_adjustment # front  right 
        legs[3].base = T * SE3(-L / 2,  W / 2, 0) * leg_adjustment # back   right


        legs[0].q = gait(qcycle, i, 0, False)
        legs[1].q = gait(qcycle, 400 - i, 100, False)
        legs[2].q = gait(qcycle, i, 200, True)
        legs[3].q = gait(qcycle, 400 - i, 300, True)
        
        # update env
        env.step(dt=0.02)
    
    return x, y, angle

max_num_cycles = 0
for cycle in range(max_num_cycles):
    if not plt.fignum_exists(env.fig.number):
        break
    
    x, y, angle = walk100mm(x, y, angle)

# walk 10cm (10 cycles)
for i in range(4):
    x, y, angle = turn1deg(x, y, angle, True)

# turn 10deg (10 cycles)
for i in range(4):
    x, y, angle = walk100mm(x, y, angle)

# walk 10cm (10 cycles)
for i in range(4):
    x, y, angle = turn1deg(x, y, angle, False)


env.hold()
plt.close('all')
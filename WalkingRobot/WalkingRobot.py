from roboticstoolbox import *
import numpy as np

from roboticstoolbox.backends.PyPlot import PyPlot
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialgeometry import Cuboid
import math

pi = math.pi
mm = 0.001

class WalkingRobot:
    def __init__(self):
        print("Initializing WalkingRobot")
        # Robot position
        self.x = 0
        self.y = 0
        self.angle = 0

        # Callback for plotting position as it walks
        self.debug_callback = None
        
        # Physical attributes
        self._width = 100 * mm
        self._length = 200 * mm

        self._Leg1 = 100 * mm
        self._Leg2 = 100 * mm
 
        # Create a leg model
        self._leg_model = ERobot(
            ET.Rz() * ET.Rx() * ET.ty(-self._Leg1) *
            ET.Rx() * ET.tz(-self._Leg2)
        )

        # Build cycles. Same animations but different animationSpeeds
        print("Building qcycles, this may take a while...")
        
        # Durations
        self._walk_duration = 0.5
        self._turn_duration = 0.5


        self._walk_qcycle = self._createGaitCycle(self._walk_duration)
        self._turn_qcycle = self._createGaitCycle(self._turn_duration)

        print("Cycles built")


        # Create environment
        self._env = PyPlot()
        size = 400 * mm
        self._env.launch(limits=[-size, size, -size, size, -0.15, 0.05])

        # Create main body
        self.body = Cuboid([self._length, self._width, 30 * mm], color='b')
        self.body.base = SE3(0, 0, 0)
        
        self._env.add(self.body)
        
        # Create legs
        self.legs = [
            ERobot(self._leg_model, name='leg0'),
            ERobot(self._leg_model, name='leg1'),
            ERobot(self._leg_model, name='leg2'),
            ERobot(self._leg_model, name='leg3')
        ]

        # Add robot legs to enviroment
        for leg in self.legs:
            leg.q = np.zeros(3)
            self._env.add(leg, readonly=True, jointaxes=False, eeframe=False, shadow=False)

        # Assemble the robot in the enviroment
        self._updatePose()
    
    def goto(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

        self._updatePose()

    def _updatePose(self):
        """
        Relates the robot to the position given by the pose parameters
        """
        # Debug callback for map visualization
        if self.debug_callback is not None:
            self.debug_callback(self.x, self.y, self.angle)
        
        T = SE3(self.x, self.y, 0) * SE3.Rz(self.angle)
        self.body.base = T

        self.leg_adjustment = SE3.Rz(pi)

        self.legs[0].base = T * SE3( self._length / 2, -self._width / 2, 0)
        self.legs[1].base = T * SE3(-self._length / 2, -self._width / 2, 0)
        self.legs[2].base = T * SE3( self._length / 2,  self._width / 2, 0) * self.leg_adjustment
        self.legs[3].base = T * SE3(-self._length / 2,  self._width / 2, 0) * self.leg_adjustment

        # Camera follows robot
        window = 0.2
        self._env.ax.set_xlim(self.x - window, self.x + window)
        self._env.ax.set_ylim(self.y - window, self.y + window)

    def _createGaitCycle(self, duration):
        """
        5-point rectangular gait cycle (Corke-style),
        but solved with iterative IK using previous q as seed.
        """

        dt = 0.01
        n_frames = int(duration / dt)

        # ---- Geometry (same idea as old code) ----
        x_f = 0.05      # 50 mm forward
        x_b = -0.05     # 50 mm backward
        y   = -0.05     # lateral offset
        z_u = -0.02     # swing height
        z_d = -0.05     # ground height

        segments = [
            [x_f, y, z_d],
            [x_b, y, z_d],
            [x_b, y, z_u],
            [x_f, y, z_u],
            [x_f, y, z_d],
        ]

        segments = np.array(segments)

        # segment durations (same structure as original PRM gait)
        seg_time = np.array([3, 0.25, 0.5, 0.25, 0])  # last closes loop
        seg_time = seg_time / seg_time.sum()

        # frames per segment
        seg_frames = (seg_time * n_frames).astype(int)

        q_cycle = []
        q_prev = np.array([0.0, 0.0, -0.8])

        for i in range(len(segments) - 1):

            p0 = segments[i]
            p1 = segments[i + 1]

            for k in range(seg_frames[i]):

                alpha = k / max(seg_frames[i], 1)

                # linear interpolation between keypoints
                x = (1 - alpha) * p0[0] + alpha * p1[0]
                y_pos = (1 - alpha) * p0[1] + alpha * p1[1]
                z = (1 - alpha) * p0[2] + alpha * p1[2]

                foot_pose = SE3(x, y_pos, z)

                sol = self._leg_model.ikine_LM(
                    foot_pose,
                    q0=q_prev,
                    mask=[1, 1, 1, 0, 0, 0]
                )

                q = sol.q
                q_cycle.append(q)
                q_prev = q

        return np.array(q_cycle)
    

    def _gait(self, cycle, elapsed, offset, reversed):
        
        elapsed = (elapsed + int(offset)) % cycle.shape[0]

        q = cycle[elapsed, :].copy()
        if reversed:
            q[0] = -q[0]   # for left-side legs
        return q

    def followPath(self, path):
        start = path[0]
        self.goto(start[0], start[1], self.angle)
        
        # Loop over every node in the path except the goal
        for x, y in path[:-1]:
            dx = x - self.x
            dy = y - self.y
            
            # Calculate numdbers of walking cycles
            dst = math.hypot(dx, dy)
            numWalks = round(dst / (100 * mm))

            # Calculate number of turning cycles
            targetAngle = math.atan2(dy, dx)
            dtheta = targetAngle - self.angle
            dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
           
            numTurns = round(np.rad2deg(dtheta))

            print(f'turns: {numTurns}, walks: {numWalks}')

            clockwise = (numTurns < 0)
            for i in range(abs(numTurns)):
                self.turn1deg(clockwise)
                print(f"Angle: {np.rad2deg(self.angle)}")

            for i in range(numWalks):
                self.walk100mm()
                print(self.x, self.y)

    def walk100mm(self):
        numFrames = self._walk_qcycle.shape[0]
        dt = self._walk_duration / numFrames

        distance = 100 * mm  # 100mm
        step_length = distance / numFrames

        for elapsed in range(numFrames):
             # Check if quit application
            if not plt.fignum_exists(self._env.fig.number):
                break

            # Basic trigonometric movement for a car like motion system (x, y, angle)
            self.x += step_length * math.cos(self.angle)
            self.y += step_length * math.sin(self.angle)

            self._updatePose()

            # Move the joints of the legs 
            self.legs[0].q = self._gait(self._walk_qcycle, elapsed, numFrames * 0 / 4, False)
            self.legs[1].q = self._gait(self._walk_qcycle, elapsed, numFrames * 1 / 4, False)
            self.legs[2].q = self._gait(self._walk_qcycle, elapsed, numFrames * 2 / 4, True)
            self.legs[3].q = self._gait(self._walk_qcycle, elapsed, numFrames * 3 / 4, True)

            self._env.step(dt= dt)

    def turn1deg(self, clockwise=False):
        numFrames = self._turn_qcycle.shape[0]
        dt = self._turn_duration / numFrames

        # Rotation velocity per frame
        turning_speed = np.deg2rad(1) # rotation angle per qcycle

        if (clockwise): 
            turning_speed *= -1
    
        rotation_step = turning_speed / numFrames

        for elapsed in range(numFrames):
            # Check if quit application
            if not plt.fignum_exists(self._env.fig.number):
                break

            self.angle += rotation_step
            
            # Skip every other frame in turn animation for perfomance reasons
            if (elapsed % 2 != 0): continue
            
            self._updatePose()

            self.legs[0].q = self._gait(self._turn_qcycle, elapsed,       numFrames * 0 / 4, False)
            self.legs[1].q = self._gait(self._turn_qcycle, numFrames - elapsed, numFrames * 1 / 4, False)
            self.legs[2].q = self._gait(self._turn_qcycle, elapsed,       numFrames * 2 / 4, True)
            self.legs[3].q = self._gait(self._turn_qcycle, numFrames - elapsed, numFrames * 3 / 4, True)
            
            # update env
            self._env.step(dt=dt)

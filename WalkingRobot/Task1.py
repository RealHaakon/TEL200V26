from WalkingRobot import WalkingRobot

# Initialize the robot
robot = WalkingRobot()

# 1) walk the robot 100 cm forward
for i in range(10):
    robot.walk100mm()

# turn the robot 10 degrees counterclockwise
for i in range(10):
     robot.turn1deg(False)

# reset robot position
robot.goto(0, 0, 0)

# 2) walk the robot 100 cm forward A->B
for i in range(10):
    robot.walk100mm()

# turn the robot 10 degrees clockwise B -> D
for i in range(10):
     robot.turn1deg(True)


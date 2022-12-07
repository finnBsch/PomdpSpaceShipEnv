import numpy as np
import matplotlib.pyplot as plt

def get_goal_v(x, y, vx, vy, goalx, goaly):
    ang_v = np.arctan2(vy, vx)
    ang_to_goal = np.arctan2((goaly-x), (goalx-y))
    d_angle = ang_v - ang_to_goal
    v = np.sqrt(vx**2 + vy**2)
    return np.cos(d_angle)*v

print(get_goal_v(0, 0, 2, 1, 3, 1))

import pomdp_spaceship_env
import scikit_build_example
import numpy as np
import time
print(pomdp_spaceship_env.__version__)
a = pomdp_spaceship_env.Config()
a.Viz = True
b = pomdp_spaceship_env.SpaceShip(a, 1)
t = time.time()
for i in range(10000):
    b.Step()
    j = b.GetState()
print(time.time() - t)

import pomdp_spaceship_env
# import librlsimpy as pomdp_spaceship_env
import numpy as np
import time
import scikit_build_example


# print(help(pomdp_spaceship_env))
print(pomdp_spaceship_env.__version__)
a = pomdp_spaceship_env.Config()
a.Viz = True
a.ResX = int(1920/2)
a.ResY = int(1080/2)
n_ships = 20
b = pomdp_spaceship_env.Env(a, n_ships)
print(b)
scikit_build_example.add(1, 2)
b.Reset()
t = time.time()
for i in range(10000):
    ins = np.array([[10, 10, -1, 1]], dtype=np.float32)
    ins = np.repeat(ins, n_ships, axis=0)
    b.Step()
    j = b.GetState()
    ins.tolist()
    j2 = b.GetReward()
    j3 = b.GetAgentDone()
    mm = b.GetMaxIn()
    mmmm = b.GetMinIn()
    b.SetControl(ins)
print(time.time() - t)

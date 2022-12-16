POMDP Space Ship Environment Documentation
===============================================

This is the documentation to the POMDP Space Ship Environment. This is a 2D environment
for a Space Ship with two individually controllable thrusters. The goal is to reach a desired goal point (static or dynamic).
The environment is filled with obstacles which can be sensed through distance sensors.
This renders the environment only partially
observable and thus the Markov Assumption is violated leading to issues with standard MLP-RL approaches.



`Example video <https://www.youtube.com/watch?v=su16NdsVE5I&ab_channel=FinnBusch>`_

`Source Code <https://github.com/finnBsch/PomdpSpaceShipEnv>`_

`Documentation <https://pomdpspaceshipenv.readthedocs.io/en/latest/>`_

Dependencies
---------------------
* `SFML <https://www.sfml-dev.org/>`_ for visualisation. Install with ``sudo apt install libsfml-dev`` for Debian-based distros.
* `Eigen3 <https://eigen.tuxfamily.org/index.php?title=Main_Page>`_ for math. Install with ``sudo apt install libeigen3-dev`` for Debian-based distros.
* `PyBind11 <https://github.com/pybind/pybind11>`_ for C++ bindings. Install with ``sudo apt install pybind11-dev`` for Debian-based distros.


Getting Started
---------------------
Build and install the module

.. code-block:: console

    # Clone the repository
    git clone https://github.com/finnBsch/PomdpSpaceShipEnv

    
    # Install the Python Module
    pip install PomdpSpaceShipEnv/


Example Usage
---------------------
Generally, the environment is to be used as shown here.

.. code-block :: python

   # Simple script to test environment in real-time
   
   import pomdp_spaceship_env
   import numpy as np
   
   conf = pomdp_spaceship_env.Config()
   
   N = 10000
   n_ships = 1
   
   # Set Config
   conf.Viz = True
   conf.AutoReset = True
   conf.ShareEnvs = False
   conf.NumObs = 60
   conf.DynamicGoals = False
   
   env = pomdp_spaceship_env.Env(conf, n_ships=n_ships)
   env.SetViz(True, True)  # Draw Rays and Obstacles
   
   # Use np.float32 as input data type.
   ins = np.array([[10, 10, -1, 1]], dtype=np.float32)
   ins = np.repeat(ins, n_ships, axis=0)
   env.SetControl(ins)
   
   # Loop the env
   for i in range(N):
       env.Step()  # could also provide a dt by calling .Step(dt=dt), useful for training.
       states = env.GetState()
       rewards = env.GetReward()
       dones = env.GetAgentDone()

Currently working on
---------------------
* Fully Customizable cost function (from Python)
* Reproducible benchmark scenarios

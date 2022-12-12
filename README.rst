POMDP Space Ship Environment Documentation
===============================================

This is the documentation to the POMDP Space Ship Environment. 



`Example video <https://www.youtube.com/watch?v=su16NdsVE5I&ab_channel=FinnBusch>`_

`Source Code <https://github.com/finnBsch/PomdpSpaceShipEnv>`_

Getting Started
---------------------
Build and install the module

.. code-block:: console

    # Clone the repository
    git clone https://github.com/finnBsch/PomdpSpaceShipEnv

    
    # Install the Python Module
    pip install PomdpSpaceShipEnv


Example Usage
---------------------
To use the environment, consider the following script


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
   env.SetViz(True, True)
   
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

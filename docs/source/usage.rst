Usage
===============================================

The Environment is operated through the ``Env`` interface class. The environment is stepped when
``Step()`` is called. Rewards and states can be retrieved by methods.

Interfacing the environment
---------------------------
All data from the environment are NumPy arrays of shape ``(dim, n_ships)``.

The results from each step are fetched with ``GetState()``, ``GetReward()`` and ``GetAgentDone()`` respectively.

Each ship's state is defined by

.. math::

    \small
    state = \begin{bmatrix} {\Delta g_x}/{w_\mathrm{sim}} & {\Delta g_y}/{w_\mathrm{sim}} & \sin (\phi) & \cos(\phi) & v_x & v_y & v_\phi & v_{x, g} & v_{y, g} & {d_{\mathrm{ray}1}}/{h_\mathrm{sim}} & \dots & {d_{\mathrm{ray}512}}/{h_\mathrm{sim}} \end{bmatrix}

where :math:`\Delta g_x` and :math:`\Delta g_y` are the distances between ship and goal point in :math:`x`and :math:`y`
direction respectively. The angle is represented by it's sine and cosine to overcome wrapping issues. The state
furthermore contains the velocities of the ship as well as the goal point and the distance readings from the 512 rays
casted.

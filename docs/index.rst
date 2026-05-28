.. image:: assets/f1_stickers_01.png
  :width: 60
  :align: left

F1TENTH Gym JAX Documentation
================================================

Overview
---------
The F1TENTH Gym JAX environment is created for research that needs a deterministic, vectorizable vehicle simulation with multiple vehicles in the same environment, with applications in reinforcement learning.

The environment is designed with determinism in mind. All agents' physics simulation are stepped simultaneously, and all randomness is controlled by explicit JAX PRNG keys. The explicit stepping API also enables ``jax.jit`` and ``jax.vmap`` workflows.

GitHub repo: https://github.com/f1tenth/f1tenth_gym_jax

Note that the GitHub repository contains the source for these docs. If you see a mistake, please contribute a fix.

Citing
--------
If you find this Gym environment useful, please consider citing:

.. code::
  
  @inproceedings{okelly2020f1tenth,
    title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
    author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
    booktitle={NeurIPS 2019 Competition and Demonstration Track},
    pages={77--89},
    year={2020},
    organization={PMLR}
  }

Physical Platform
-------------------

To build a physical 1/10th scale vehicle, follow the guide here:
https://roboracer.ai/build

.. image:: https://f1tenth.readthedocs.io/en/foxy_test/_images/f1tenth_NX.png
  :width: 400
  :align: center
  :target: https://roboracer.ai/build

.. toctree::
  :caption: INSTALLATION
  :maxdepth: 2

  installation


.. toctree::
  :caption: USAGE
  :maxdepth: 2

  basic_usage
  customized_usage
  visualization

.. toctree::
  :caption: MODEL REFERENCE
  :maxdepth: 2

  dynamics

.. toctree::
  :caption: REPRODUCIBILITY
  :maxdepth: 2

  reproduce

.. toctree::
  :caption: API REFERENCE
  :maxdepth: 2

  api/index

![CI](https://github.com/f1tenth/f1tenth_gym_jax/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym_jax/actions/workflows/docker.yml/badge.svg)
![Lint](https://github.com/f1tenth/f1tenth_gym_jax/actions/workflows/lint.yml/badge.svg)

# F1TENTH Gym JAX

This repository contains a JAX-compatible multi-agent F1TENTH racing environment.
The main API is `f1tenth_gym_jax.make(...)`, which returns a jittable environment
with `reset(key)` and `step(key, state, actions)` methods.

The project is under active development.

## Quickstart

Install the package in an isolated environment:

```bash
git clone https://github.com/f1tenth/f1tenth_gym_jax.git
cd f1tenth_gym_jax
uv sync
```

Optional extras are split by workflow:

```bash
uv sync --extra examples  # plotting, video, and track generation examples
uv sync --extra rl        # PPO training/evaluation dependencies
uv sync --extra docs      # Sphinx documentation build
```

Run a minimal rollout:

```python
import jax
import jax.numpy as jnp

from f1tenth_gym_jax import make

env = make("Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_100_v0")
key = jax.random.key(0)
obs, state = env.reset(key)

actions = {"agent_0": jnp.array([0.0, 1.0])}
key, step_key = jax.random.split(key)
obs, state, rewards, dones, infos = env.step(step_key, state, actions)
```

Longer example usage lives in:

- `examples/train_ppo_example.py`
- `examples/eval_ppo_example.py`
- `examples/waypoint_follow.py`
- `examples/mppi_example.py`

## Environment IDs

Environment IDs use this pattern:

```text
{map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_{max_steps}_v0
```

For the default episode length, use the shorthand form without the
`max_steps` field:

```text
{map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_v0
```

Examples:

```text
Spielberg_1_scan_collision_progress+alive_velocity+steeringangle_10_v0
Spielberg_1_scan_collision_progress+alive_velocity+steeringangle_10_500_v0
```

The final `v0` is the environment ID version.
Older model filenames that omit `timestep_ratio` are still accepted and use
`timestep_ratio=1`.

Bundled maps are loaded from the installed package. Downloaded maps are cached
under `$XDG_CACHE_HOME/f1tenth_gym_jax/maps` by default; set
`F1TENTH_GYM_JAX_MAP_DIR` to use another writable map directory.

## Docker

```bash
docker build -t f1tenth_gym_jax -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_jax
```

## Citing

If you find this environment useful, please consider citing:

```text
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```

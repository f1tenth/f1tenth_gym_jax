import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.utils import batchify, unbatchify
from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer

@jax.jit
def pure_pursuit_2(pose, wpts, Ld=3.0, L=0.33, dt=0.1):
    x, y, theta, current_steer, current_v = pose
    dx, dy = wpts[:, 0] - x, wpts[:, 1] - y
    xv = dx * jnp.cos(theta) + dy * jnp.sin(theta)
    yv = -dx * jnp.sin(theta) + dy * jnp.cos(theta)

    ds = jnp.hypot(xv, yv)
    i = jnp.argmin(jnp.abs(ds - Ld))
    x_ld, y_ld = xv[i], yv[i]

    kappa = 2. * y_ld / (Ld ** 2)
    desired_steer = jnp.arctan(L * kappa)

    sv = (desired_steer - current_steer) / dt
    sv = jnp.clip(sv, -3.2, 3.2)

    desired_v = wpts[i, 2]
    accl = (desired_v - current_v) / dt
    accl = jnp.clip(accl, -10.0, 10.0)

    return jnp.array([sv, accl])

def main():

    num_agents = 3
    num_envs = 10
    num_actors = num_agents * num_envs

    env = make(f"Spielberg_{num_agents}_noscan_time_v0")
    l = env.track.raceline
    waypoints = jnp.vstack((l.xs, l.ys, l.vxs)).T
    rng = jax.random.key(0)

    @jax.jit
    def env_init(rng):
        rng, _rng = jax.random.split(rng)
        reset_rngs = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)
        return (env_state, obsv, rng)

    @jax.jit
    def env_step(runner_state, unused):
        env_state, last_obsv, rng = runner_state
        rng, _rng = jax.random.split(rng)
        step_rngs = jax.random.split(_rng, num_envs)

        batched_obs = batchify(last_obsv, env.agents, num_actors)
        batched_actions = jax.vmap(pure_pursuit_2, in_axes=(0, None))(
            batched_obs[:, [0, 1, 4, 2, 3]], waypoints
        )
        env_actions = unbatchify(batched_actions, env.agents, num_envs, num_agents)

        obsv, env_state, _, _, info = jax.vmap(env.step)(
            step_rngs, env_state, env_actions
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, runner_state

    final_runner, all_runner_state = jax.lax.scan(env_step, env_init(rng), length=1000)

    player = TrajRenderer(env)
    player.render(np.array(all_runner_state[0].cartesian_states))
    print(jnp.any(all_runner_state[0].collisions))


if __name__ == "__main__":
    main()
